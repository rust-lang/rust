use monomorphize::Instance;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::session::config::OptLevel;
use rustc::ty::{self, Ty, TyCtxt, ClosureSubsts, GeneratorSubsts};
use rustc::ty::subst::Substs;
use syntax::ast;
use syntax::attr::InlineAttr;
use std::fmt::{self, Write};
use std::iter;
use rustc::mir::mono::Linkage;
use syntax_pos::symbol::Symbol;
use syntax::source_map::Span;
pub use rustc::mir::mono::MonoItem;

/// Describes how a monomorphization will be instantiated in object files.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum InstantiationMode {
    /// There will be exactly one instance of the given MonoItem. It will have
    /// external linkage so that it can be linked to from other codegen units.
    GloballyShared {
        /// In some compilation scenarios we may decide to take functions that
        /// are typically `LocalCopy` and instead move them to `GloballyShared`
        /// to avoid codegenning them a bunch of times. In this situation,
        /// however, our local copy may conflict with other crates also
        /// inlining the same function.
        ///
        /// This flag indicates that this situation is occurring, and informs
        /// symbol name calculation that some extra mangling is needed to
        /// avoid conflicts. Note that this may eventually go away entirely if
        /// ThinLTO enables us to *always* have a globally shared instance of a
        /// function within one crate's compilation.
        may_conflict: bool,
    },

    /// Each codegen unit containing a reference to the given MonoItem will
    /// have its own private copy of the function (with internal linkage).
    LocalCopy,
}

pub trait MonoItemExt<'a, 'tcx>: fmt::Debug {
    fn as_mono_item(&self) -> &MonoItem<'tcx>;

    fn is_generic_fn(&self) -> bool {
        match *self.as_mono_item() {
            MonoItem::Fn(ref instance) => {
                instance.substs.types().next().is_some()
            }
            MonoItem::Static(..) |
            MonoItem::GlobalAsm(..) => false,
        }
    }

    fn symbol_name(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> ty::SymbolName {
        match *self.as_mono_item() {
            MonoItem::Fn(instance) => tcx.symbol_name(instance),
            MonoItem::Static(def_id) => {
                tcx.symbol_name(Instance::mono(tcx, def_id))
            }
            MonoItem::GlobalAsm(node_id) => {
                let def_id = tcx.hir().local_def_id(node_id);
                ty::SymbolName {
                    name: Symbol::intern(&format!("global_asm_{:?}", def_id)).as_interned_str()
                }
            }
        }
    }
    fn instantiation_mode(&self,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>)
                          -> InstantiationMode {
        let inline_in_all_cgus =
            tcx.sess.opts.debugging_opts.inline_in_all_cgus.unwrap_or_else(|| {
                tcx.sess.opts.optimize != OptLevel::No
            }) && !tcx.sess.opts.cg.link_dead_code;

        match *self.as_mono_item() {
            MonoItem::Fn(ref instance) => {
                let entry_def_id =
                    tcx.sess.entry_fn.borrow().map(|(id, _, _)| tcx.hir().local_def_id(id));
                // If this function isn't inlined or otherwise has explicit
                // linkage, then we'll be creating a globally shared version.
                if self.explicit_linkage(tcx).is_some() ||
                    !instance.def.requires_local(tcx) ||
                    Some(instance.def_id()) == entry_def_id
                {
                    return InstantiationMode::GloballyShared  { may_conflict: false }
                }

                // At this point we don't have explicit linkage and we're an
                // inlined function. If we're inlining into all CGUs then we'll
                // be creating a local copy per CGU
                if inline_in_all_cgus {
                    return InstantiationMode::LocalCopy
                }

                // Finally, if this is `#[inline(always)]` we're sure to respect
                // that with an inline copy per CGU, but otherwise we'll be
                // creating one copy of this `#[inline]` function which may
                // conflict with upstream crates as it could be an exported
                // symbol.
                match tcx.codegen_fn_attrs(instance.def_id()).inline {
                    InlineAttr::Always => InstantiationMode::LocalCopy,
                    _ => {
                        InstantiationMode::GloballyShared  { may_conflict: true }
                    }
                }
            }
            MonoItem::Static(..) |
            MonoItem::GlobalAsm(..) => {
                InstantiationMode::GloballyShared { may_conflict: false }
            }
        }
    }

    fn explicit_linkage(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Option<Linkage> {
        let def_id = match *self.as_mono_item() {
            MonoItem::Fn(ref instance) => instance.def_id(),
            MonoItem::Static(def_id) => def_id,
            MonoItem::GlobalAsm(..) => return None,
        };

        let codegen_fn_attrs = tcx.codegen_fn_attrs(def_id);
        codegen_fn_attrs.linkage
    }

    /// Returns whether this instance is instantiable - whether it has no unsatisfied
    /// predicates.
    ///
    /// In order to codegen an item, all of its predicates must hold, because
    /// otherwise the item does not make sense. Type-checking ensures that
    /// the predicates of every item that is *used by* a valid item *do*
    /// hold, so we can rely on that.
    ///
    /// However, we codegen collector roots (reachable items) and functions
    /// in vtables when they are seen, even if they are not used, and so they
    /// might not be instantiable. For example, a programmer can define this
    /// public function:
    ///
    ///     pub fn foo<'a>(s: &'a mut ()) where &'a mut (): Clone {
    ///         <&mut () as Clone>::clone(&s);
    ///     }
    ///
    /// That function can't be codegened, because the method `<&mut () as Clone>::clone`
    /// does not exist. Luckily for us, that function can't ever be used,
    /// because that would require for `&'a mut (): Clone` to hold, so we
    /// can just not emit any code, or even a linker reference for it.
    ///
    /// Similarly, if a vtable method has such a signature, and therefore can't
    /// be used, we can just not emit it and have a placeholder (a null pointer,
    /// which will never be accessed) in its place.
    fn is_instantiable(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> bool {
        debug!("is_instantiable({:?})", self);
        let (def_id, substs) = match *self.as_mono_item() {
            MonoItem::Fn(ref instance) => (instance.def_id(), instance.substs),
            MonoItem::Static(def_id) => (def_id, Substs::empty()),
            // global asm never has predicates
            MonoItem::GlobalAsm(..) => return true
        };

        tcx.substitute_normalize_and_test_predicates((def_id, &substs))
    }

    fn to_string(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> String {
        return match *self.as_mono_item() {
            MonoItem::Fn(instance) => {
                to_string_internal(tcx, "fn ", instance)
            },
            MonoItem::Static(def_id) => {
                let instance = Instance::new(def_id, tcx.intern_substs(&[]));
                to_string_internal(tcx, "static ", instance)
            },
            MonoItem::GlobalAsm(..) => {
                "global_asm".to_string()
            }
        };

        fn to_string_internal<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        prefix: &str,
                                        instance: Instance<'tcx>)
                                        -> String {
            let mut result = String::with_capacity(32);
            result.push_str(prefix);
            let printer = DefPathBasedNames::new(tcx, false, false);
            printer.push_instance_as_string(instance, &mut result);
            result
        }
    }

    fn local_span(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Option<Span> {
        match *self.as_mono_item() {
            MonoItem::Fn(Instance { def, .. }) => {
                tcx.hir().as_local_node_id(def.def_id())
            }
            MonoItem::Static(def_id) => {
                tcx.hir().as_local_node_id(def_id)
            }
            MonoItem::GlobalAsm(node_id) => {
                Some(node_id)
            }
        }.map(|node_id| tcx.hir().span(node_id))
    }
}

impl<'a, 'tcx> MonoItemExt<'a, 'tcx> for MonoItem<'tcx> {
    fn as_mono_item(&self) -> &MonoItem<'tcx> {
        self
    }
}

//=-----------------------------------------------------------------------------
// MonoItem String Keys
//=-----------------------------------------------------------------------------

// The code below allows for producing a unique string key for a mono item.
// These keys are used by the handwritten auto-tests, so they need to be
// predictable and human-readable.
//
// Note: A lot of this could looks very similar to what's already in the
//       ppaux module. It would be good to refactor things so we only have one
//       parameterizable implementation for printing types.

/// Same as `unique_type_name()` but with the result pushed onto the given
/// `output` parameter.
pub struct DefPathBasedNames<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    omit_disambiguators: bool,
    omit_local_crate_name: bool,
}

impl<'a, 'tcx> DefPathBasedNames<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>,
               omit_disambiguators: bool,
               omit_local_crate_name: bool)
               -> Self {
        DefPathBasedNames {
            tcx,
            omit_disambiguators,
            omit_local_crate_name,
        }
    }

    pub fn push_type_name(&self, t: Ty<'tcx>, output: &mut String) {
        match t.sty {
            ty::Bool              => output.push_str("bool"),
            ty::Char              => output.push_str("char"),
            ty::Str               => output.push_str("str"),
            ty::Never             => output.push_str("!"),
            ty::Int(ast::IntTy::Isize)    => output.push_str("isize"),
            ty::Int(ast::IntTy::I8)    => output.push_str("i8"),
            ty::Int(ast::IntTy::I16)   => output.push_str("i16"),
            ty::Int(ast::IntTy::I32)   => output.push_str("i32"),
            ty::Int(ast::IntTy::I64)   => output.push_str("i64"),
            ty::Int(ast::IntTy::I128)   => output.push_str("i128"),
            ty::Uint(ast::UintTy::Usize)   => output.push_str("usize"),
            ty::Uint(ast::UintTy::U8)   => output.push_str("u8"),
            ty::Uint(ast::UintTy::U16)  => output.push_str("u16"),
            ty::Uint(ast::UintTy::U32)  => output.push_str("u32"),
            ty::Uint(ast::UintTy::U64)  => output.push_str("u64"),
            ty::Uint(ast::UintTy::U128)  => output.push_str("u128"),
            ty::Float(ast::FloatTy::F32) => output.push_str("f32"),
            ty::Float(ast::FloatTy::F64) => output.push_str("f64"),
            ty::Adt(adt_def, substs) => {
                self.push_def_path(adt_def.did, output);
                self.push_type_params(substs, iter::empty(), output);
            },
            ty::Tuple(component_types) => {
                output.push('(');
                for &component_type in component_types {
                    self.push_type_name(component_type, output);
                    output.push_str(", ");
                }
                if !component_types.is_empty() {
                    output.pop();
                    output.pop();
                }
                output.push(')');
            },
            ty::RawPtr(ty::TypeAndMut { ty: inner_type, mutbl } ) => {
                output.push('*');
                match mutbl {
                    hir::MutImmutable => output.push_str("const "),
                    hir::MutMutable => output.push_str("mut "),
                }

                self.push_type_name(inner_type, output);
            },
            ty::Ref(_, inner_type, mutbl) => {
                output.push('&');
                if mutbl == hir::MutMutable {
                    output.push_str("mut ");
                }

                self.push_type_name(inner_type, output);
            },
            ty::Array(inner_type, len) => {
                output.push('[');
                self.push_type_name(inner_type, output);
                write!(output, "; {}", len.unwrap_usize(self.tcx)).unwrap();
                output.push(']');
            },
            ty::Slice(inner_type) => {
                output.push('[');
                self.push_type_name(inner_type, output);
                output.push(']');
            },
            ty::Dynamic(ref trait_data, ..) => {
                let principal = trait_data.principal();
                self.push_def_path(principal.def_id(), output);
                self.push_type_params(
                    principal.skip_binder().substs,
                    trait_data.projection_bounds(),
                    output,
                );
            },
            ty::Foreign(did) => self.push_def_path(did, output),
            ty::FnDef(..) |
            ty::FnPtr(_) => {
                let sig = t.fn_sig(self.tcx);
                if sig.unsafety() == hir::Unsafety::Unsafe {
                    output.push_str("unsafe ");
                }

                let abi = sig.abi();
                if abi != ::rustc_target::spec::abi::Abi::Rust {
                    output.push_str("extern \"");
                    output.push_str(abi.name());
                    output.push_str("\" ");
                }

                output.push_str("fn(");

                let sig = self.tcx.normalize_erasing_late_bound_regions(
                    ty::ParamEnv::reveal_all(),
                    &sig,
                );

                if !sig.inputs().is_empty() {
                    for &parameter_type in sig.inputs() {
                        self.push_type_name(parameter_type, output);
                        output.push_str(", ");
                    }
                    output.pop();
                    output.pop();
                }

                if sig.variadic {
                    if !sig.inputs().is_empty() {
                        output.push_str(", ...");
                    } else {
                        output.push_str("...");
                    }
                }

                output.push(')');

                if !sig.output().is_unit() {
                    output.push_str(" -> ");
                    self.push_type_name(sig.output(), output);
                }
            },
            ty::Generator(def_id, GeneratorSubsts { ref substs }, _) |
            ty::Closure(def_id, ClosureSubsts { ref substs }) => {
                self.push_def_path(def_id, output);
                let generics = self.tcx.generics_of(self.tcx.closure_base_def_id(def_id));
                let substs = substs.truncate_to(self.tcx, generics);
                self.push_type_params(substs, iter::empty(), output);
            }
            ty::Error |
            ty::Bound(..) |
            ty::Infer(_) |
            ty::Placeholder(..) |
            ty::UnnormalizedProjection(..) |
            ty::Projection(..) |
            ty::Param(_) |
            ty::GeneratorWitness(_) |
            ty::Opaque(..) => {
                bug!("DefPathBasedNames: Trying to create type name for \
                                         unexpected type: {:?}", t);
            }
        }
    }

    pub fn push_def_path(&self,
                         def_id: DefId,
                         output: &mut String) {
        let def_path = self.tcx.def_path(def_id);

        // some_crate::
        if !(self.omit_local_crate_name && def_id.is_local()) {
            output.push_str(&self.tcx.crate_name(def_path.krate).as_str());
            output.push_str("::");
        }

        // foo::bar::ItemName::
        for part in self.tcx.def_path(def_id).data {
            if self.omit_disambiguators {
                write!(output, "{}::", part.data.as_interned_str()).unwrap();
            } else {
                write!(output, "{}[{}]::",
                       part.data.as_interned_str(),
                       part.disambiguator).unwrap();
            }
        }

        // remove final "::"
        output.pop();
        output.pop();
    }

    fn push_type_params<I>(&self,
                            substs: &Substs<'tcx>,
                            projections: I,
                            output: &mut String)
        where I: Iterator<Item=ty::PolyExistentialProjection<'tcx>>
    {
        let mut projections = projections.peekable();
        if substs.types().next().is_none() && projections.peek().is_none() {
            return;
        }

        output.push('<');

        for type_parameter in substs.types() {
            self.push_type_name(type_parameter, output);
            output.push_str(", ");
        }

        for projection in projections {
            let projection = projection.skip_binder();
            let name = &self.tcx.associated_item(projection.item_def_id).ident.as_str();
            output.push_str(name);
            output.push_str("=");
            self.push_type_name(projection.ty, output);
            output.push_str(", ");
        }

        output.pop();
        output.pop();

        output.push('>');
    }

    pub fn push_instance_as_string(&self,
                                   instance: Instance<'tcx>,
                                   output: &mut String) {
        self.push_def_path(instance.def_id(), output);
        self.push_type_params(instance.substs, iter::empty(), output);
    }
}
