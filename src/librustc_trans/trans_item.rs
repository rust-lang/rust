use base::llvm_linkage_by_name;
use glue::DropGlueKind;
use llvm;
use monomorphize::Instance;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst;
use std::hash::{Hash, Hasher};
use syntax::ast::{self, NodeId};
use syntax::attr;
use syntax::parse::token;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum TransItem<'tcx> {
    DropGlue(DropGlueKind<'tcx>),
    Fn(Instance<'tcx>),
    Static(NodeId)
}

impl<'tcx> Hash for TransItem<'tcx> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        match *self {
            TransItem::DropGlue(t) => {
                0u8.hash(s);
                t.hash(s);
            },
            TransItem::Fn(instance) => {
                1u8.hash(s);
                instance.def.hash(s);
                (instance.substs as *const _ as usize).hash(s);
            }
            TransItem::Static(node_id) => {
                2u8.hash(s);
                node_id.hash(s);
            }
        };
    }
}

//=-----------------------------------------------------------------------------
// TransItem String Keys
//=-----------------------------------------------------------------------------

// The code below allows for producing a unique string key for a trans item.
// These keys are used by the handwritten auto-tests, so they need to be
// predictable and human-readable.
//
// Note: A lot of this could looks very similar to what's already in the
//       ppaux module. It would be good to refactor things so we only have one
//       parameterizable implementation for printing types.

/// Same as `unique_type_name()` but with the result pushed onto the given
/// `output` parameter.
pub fn push_unique_type_name<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       t: ty::Ty<'tcx>,
                                       output: &mut String) {
    match t.sty {
        ty::TyBool              => output.push_str("bool"),
        ty::TyChar              => output.push_str("char"),
        ty::TyStr               => output.push_str("str"),
        ty::TyInt(ast::IntTy::Is)    => output.push_str("isize"),
        ty::TyInt(ast::IntTy::I8)    => output.push_str("i8"),
        ty::TyInt(ast::IntTy::I16)   => output.push_str("i16"),
        ty::TyInt(ast::IntTy::I32)   => output.push_str("i32"),
        ty::TyInt(ast::IntTy::I64)   => output.push_str("i64"),
        ty::TyUint(ast::UintTy::Us)   => output.push_str("usize"),
        ty::TyUint(ast::UintTy::U8)   => output.push_str("u8"),
        ty::TyUint(ast::UintTy::U16)  => output.push_str("u16"),
        ty::TyUint(ast::UintTy::U32)  => output.push_str("u32"),
        ty::TyUint(ast::UintTy::U64)  => output.push_str("u64"),
        ty::TyFloat(ast::FloatTy::F32) => output.push_str("f32"),
        ty::TyFloat(ast::FloatTy::F64) => output.push_str("f64"),
        ty::TyStruct(adt_def, substs) |
        ty::TyEnum(adt_def, substs) => {
            push_item_name(tcx, adt_def.did, output);
            push_type_params(tcx, &substs.types, &[], output);
        },
        ty::TyTuple(component_types) => {
            output.push('(');
            for &component_type in component_types {
                push_unique_type_name(tcx, component_type, output);
                output.push_str(", ");
            }
            if !component_types.is_empty() {
                output.pop();
                output.pop();
            }
            output.push(')');
        },
        ty::TyBox(inner_type) => {
            output.push_str("Box<");
            push_unique_type_name(tcx, inner_type, output);
            output.push('>');
        },
        ty::TyRawPtr(ty::TypeAndMut { ty: inner_type, mutbl } ) => {
            output.push('*');
            match mutbl {
                hir::MutImmutable => output.push_str("const "),
                hir::MutMutable => output.push_str("mut "),
            }

            push_unique_type_name(tcx, inner_type, output);
        },
        ty::TyRef(_, ty::TypeAndMut { ty: inner_type, mutbl }) => {
            output.push('&');
            if mutbl == hir::MutMutable {
                output.push_str("mut ");
            }

            push_unique_type_name(tcx, inner_type, output);
        },
        ty::TyArray(inner_type, len) => {
            output.push('[');
            push_unique_type_name(tcx, inner_type, output);
            output.push_str(&format!("; {}", len));
            output.push(']');
        },
        ty::TySlice(inner_type) => {
            output.push('[');
            push_unique_type_name(tcx, inner_type, output);
            output.push(']');
        },
        ty::TyTrait(ref trait_data) => {
            push_item_name(tcx, trait_data.principal.skip_binder().def_id, output);
            push_type_params(tcx,
                             &trait_data.principal.skip_binder().substs.types,
                             &trait_data.bounds.projection_bounds,
                             output);
        },
        ty::TyFnDef(_, _, &ty::BareFnTy{ unsafety, abi, ref sig } ) |
        ty::TyFnPtr(&ty::BareFnTy{ unsafety, abi, ref sig } ) => {
            if unsafety == hir::Unsafety::Unsafe {
                output.push_str("unsafe ");
            }

            if abi != ::abi::Abi::Rust {
                output.push_str("extern \"");
                output.push_str(abi.name());
                output.push_str("\" ");
            }

            output.push_str("fn(");

            let sig = tcx.erase_late_bound_regions(sig);
            if !sig.inputs.is_empty() {
                for &parameter_type in &sig.inputs {
                    push_unique_type_name(tcx, parameter_type, output);
                    output.push_str(", ");
                }
                output.pop();
                output.pop();
            }

            if sig.variadic {
                if !sig.inputs.is_empty() {
                    output.push_str(", ...");
                } else {
                    output.push_str("...");
                }
            }

            output.push(')');

            match sig.output {
                ty::FnConverging(result_type) if result_type.is_nil() => {}
                ty::FnConverging(result_type) => {
                    output.push_str(" -> ");
                    push_unique_type_name(tcx, result_type, output);
                }
                ty::FnDiverging => {
                    output.push_str(" -> !");
                }
            }
        },
        ty::TyClosure(def_id, ref closure_substs) => {
            push_item_name(tcx, def_id, output);
            output.push_str("{");
            output.push_str(&format!("{}:{}", def_id.krate, def_id.index.as_usize()));
            output.push_str("}");
            push_type_params(tcx, &closure_substs.func_substs.types, &[], output);
        }
        ty::TyError |
        ty::TyInfer(_) |
        ty::TyProjection(..) |
        ty::TyParam(_) => {
            bug!("debuginfo: Trying to create type name for \
                  unexpected type: {:?}", t);
        }
    }
}

fn push_item_name(tcx: TyCtxt,
                  def_id: DefId,
                  output: &mut String) {
    let def_path = tcx.def_path(def_id);

    // some_crate::
    output.push_str(&tcx.crate_name(def_path.krate));
    output.push_str("::");

    // foo::bar::ItemName::
    for part in tcx.def_path(def_id).data {
        output.push_str(&format!("{}[{}]::",
                        part.data.as_interned_str(),
                        part.disambiguator));
    }

    // remove final "::"
    output.pop();
    output.pop();
}

fn push_type_params<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              types: &'tcx subst::VecPerParamSpace<Ty<'tcx>>,
                              projections: &[ty::PolyProjectionPredicate<'tcx>],
                              output: &mut String) {
    if types.is_empty() && projections.is_empty() {
        return;
    }

    output.push('<');

    for &type_parameter in types {
        push_unique_type_name(tcx, type_parameter, output);
        output.push_str(", ");
    }

    for projection in projections {
        let projection = projection.skip_binder();
        let name = token::get_ident_interner().get(projection.projection_ty.item_name);
        output.push_str(&name[..]);
        output.push_str("=");
        push_unique_type_name(tcx, projection.ty, output);
        output.push_str(", ");
    }

    output.pop();
    output.pop();

    output.push('>');
}

fn push_instance_as_string<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     instance: Instance<'tcx>,
                                     output: &mut String) {
    push_item_name(tcx, instance.def, output);
    push_type_params(tcx, &instance.substs.types, &[], output);
}

pub fn def_id_to_string(tcx: TyCtxt, def_id: DefId) -> String {
    let mut output = String::new();
    push_item_name(tcx, def_id, &mut output);
    output
}

pub fn type_to_string<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                ty: ty::Ty<'tcx>)
                                -> String {
    let mut output = String::new();
    push_unique_type_name(tcx, ty, &mut output);
    output
}

impl<'tcx> TransItem<'tcx> {

    pub fn requests_inline<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> bool {
        match *self {
            TransItem::Fn(ref instance) => {
                let attributes = tcx.get_attrs(instance.def);
                attr::requests_inline(&attributes[..])
            }
            TransItem::DropGlue(..) => true,
            TransItem::Static(..)   => false,
        }
    }

    pub fn is_from_extern_crate(&self) -> bool {
        match *self {
            TransItem::Fn(ref instance) => !instance.def.is_local(),
            TransItem::DropGlue(..) |
            TransItem::Static(..)   => false,
        }
    }

    pub fn is_lazily_instantiated(&self) -> bool {
        match *self {
            TransItem::Fn(ref instance) => !instance.substs.types.is_empty(),
            TransItem::DropGlue(..) => true,
            TransItem::Static(..)   => false,
        }
    }

    pub fn explicit_linkage<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Option<llvm::Linkage> {
        let def_id = match *self {
            TransItem::Fn(ref instance) => instance.def,
            TransItem::Static(node_id) => tcx.map.local_def_id(node_id),
            TransItem::DropGlue(..) => return None,
        };

        let attributes = tcx.get_attrs(def_id);
        if let Some(name) = attr::first_attr_value_str_by_name(&attributes, "linkage") {
            if let Some(linkage) = llvm_linkage_by_name(&name) {
                Some(linkage)
            } else {
                let span = tcx.map.span_if_local(def_id);
                if let Some(span) = span {
                    tcx.sess.span_fatal(span, "invalid linkage specified")
                } else {
                    tcx.sess.fatal(&format!("invalid linkage specified: {}", name))
                }
            }
        } else {
            None
        }
    }

    pub fn to_string<'a>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> String {
        let hir_map = &tcx.map;

        return match *self {
            TransItem::DropGlue(dg) => {
                let mut s = String::with_capacity(32);
                match dg {
                    DropGlueKind::Ty(_) => s.push_str("drop-glue "),
                    DropGlueKind::TyContents(_) => s.push_str("drop-glue-contents "),
                };
                push_unique_type_name(tcx, dg.ty(), &mut s);
                s
            }
            TransItem::Fn(instance) => {
                to_string_internal(tcx, "fn ", instance)
            },
            TransItem::Static(node_id) => {
                let def_id = hir_map.local_def_id(node_id);
                let instance = Instance::mono(tcx, def_id);
                to_string_internal(tcx, "static ", instance)
            },
        };

        fn to_string_internal<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        prefix: &str,
                                        instance: Instance<'tcx>)
                                        -> String {
            let mut result = String::with_capacity(32);
            result.push_str(prefix);
            push_instance_as_string(tcx, instance, &mut result);
            result
        }
    }

    pub fn to_raw_string(&self) -> String {
        match *self {
            TransItem::DropGlue(dg) => {
                format!("DropGlue({})", dg.ty() as *const _ as usize)
            }
            TransItem::Fn(instance) => {
                format!("Fn({:?}, {})",
                         instance.def,
                         instance.substs as *const _ as usize)
            }
            TransItem::Static(id) => {
                format!("Static({:?})", id)
            }
        }
    }
}
