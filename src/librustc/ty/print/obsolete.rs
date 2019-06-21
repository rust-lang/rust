//! Allows for producing a unique string key for a mono item.
//! These keys are used by the handwritten auto-tests, so they need to be
//! predictable and human-readable.
//!
//! Note: A lot of this could looks very similar to what's already in `ty::print`.
//! FIXME(eddyb) implement a custom `PrettyPrinter` for this.

use rustc::hir::def_id::DefId;
use rustc::mir::interpret::ConstValue;
use rustc::ty::subst::SubstsRef;
use rustc::ty::{self, ClosureSubsts, Const, GeneratorSubsts, Instance, Ty, TyCtxt};
use rustc::{bug, hir};
use std::fmt::Write;
use std::iter;
use syntax::ast;

/// Same as `unique_type_name()` but with the result pushed onto the given
/// `output` parameter.
pub struct DefPathBasedNames<'tcx> {
    tcx: TyCtxt<'tcx>,
    omit_disambiguators: bool,
    omit_local_crate_name: bool,
}

impl DefPathBasedNames<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, omit_disambiguators: bool, omit_local_crate_name: bool) -> Self {
        DefPathBasedNames { tcx, omit_disambiguators, omit_local_crate_name }
    }

    // Pushes the type name of the specified type to the provided string.
    // If `debug` is true, printing normally unprintable types is allowed
    // (e.g. `ty::GeneratorWitness`). This parameter should only be set when
    // this method is being used for logging purposes (e.g. with `debug!` or `info!`)
    // When being used for codegen purposes, `debug` should be set to `false`
    // in order to catch unexpected types that should never end up in a type name.
    pub fn push_type_name(&self, t: Ty<'tcx>, output: &mut String, debug: bool) {
        match t.sty {
            ty::Bool => output.push_str("bool"),
            ty::Char => output.push_str("char"),
            ty::Str => output.push_str("str"),
            ty::Never => output.push_str("!"),
            ty::Int(ast::IntTy::Isize) => output.push_str("isize"),
            ty::Int(ast::IntTy::I8) => output.push_str("i8"),
            ty::Int(ast::IntTy::I16) => output.push_str("i16"),
            ty::Int(ast::IntTy::I32) => output.push_str("i32"),
            ty::Int(ast::IntTy::I64) => output.push_str("i64"),
            ty::Int(ast::IntTy::I128) => output.push_str("i128"),
            ty::Uint(ast::UintTy::Usize) => output.push_str("usize"),
            ty::Uint(ast::UintTy::U8) => output.push_str("u8"),
            ty::Uint(ast::UintTy::U16) => output.push_str("u16"),
            ty::Uint(ast::UintTy::U32) => output.push_str("u32"),
            ty::Uint(ast::UintTy::U64) => output.push_str("u64"),
            ty::Uint(ast::UintTy::U128) => output.push_str("u128"),
            ty::Float(ast::FloatTy::F32) => output.push_str("f32"),
            ty::Float(ast::FloatTy::F64) => output.push_str("f64"),
            ty::Adt(adt_def, substs) => {
                self.push_def_path(adt_def.did, output);
                self.push_generic_params(substs, iter::empty(), output, debug);
            }
            ty::Tuple(component_types) => {
                output.push('(');
                for &component_type in component_types {
                    self.push_type_name(component_type.expect_ty(), output, debug);
                    output.push_str(", ");
                }
                if !component_types.is_empty() {
                    output.pop();
                    output.pop();
                }
                output.push(')');
            }
            ty::RawPtr(ty::TypeAndMut { ty: inner_type, mutbl }) => {
                output.push('*');
                match mutbl {
                    hir::MutImmutable => output.push_str("const "),
                    hir::MutMutable => output.push_str("mut "),
                }

                self.push_type_name(inner_type, output, debug);
            }
            ty::Ref(_, inner_type, mutbl) => {
                output.push('&');
                if mutbl == hir::MutMutable {
                    output.push_str("mut ");
                }

                self.push_type_name(inner_type, output, debug);
            }
            ty::Array(inner_type, len) => {
                output.push('[');
                self.push_type_name(inner_type, output, debug);
                write!(output, "; {}", len.unwrap_usize(self.tcx)).unwrap();
                output.push(']');
            }
            ty::Slice(inner_type) => {
                output.push('[');
                self.push_type_name(inner_type, output, debug);
                output.push(']');
            }
            ty::Dynamic(ref trait_data, ..) => {
                if let Some(principal) = trait_data.principal() {
                    self.push_def_path(principal.def_id(), output);
                    self.push_generic_params(
                        principal.skip_binder().substs,
                        trait_data.projection_bounds(),
                        output,
                        debug,
                    );
                } else {
                    output.push_str("dyn '_");
                }
            }
            ty::Foreign(did) => self.push_def_path(did, output),
            ty::FnDef(..) | ty::FnPtr(_) => {
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

                let sig =
                    self.tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);

                if !sig.inputs().is_empty() {
                    for &parameter_type in sig.inputs() {
                        self.push_type_name(parameter_type, output, debug);
                        output.push_str(", ");
                    }
                    output.pop();
                    output.pop();
                }

                if sig.c_variadic {
                    if !sig.inputs().is_empty() {
                        output.push_str(", ...");
                    } else {
                        output.push_str("...");
                    }
                }

                output.push(')');

                if !sig.output().is_unit() {
                    output.push_str(" -> ");
                    self.push_type_name(sig.output(), output, debug);
                }
            }
            ty::Generator(def_id, GeneratorSubsts { ref substs }, _)
            | ty::Closure(def_id, ClosureSubsts { ref substs }) => {
                self.push_def_path(def_id, output);
                let generics = self.tcx.generics_of(self.tcx.closure_base_def_id(def_id));
                let substs = substs.truncate_to(self.tcx, generics);
                self.push_generic_params(substs, iter::empty(), output, debug);
            }
            ty::Error
            | ty::Bound(..)
            | ty::Infer(_)
            | ty::Placeholder(..)
            | ty::UnnormalizedProjection(..)
            | ty::Projection(..)
            | ty::Param(_)
            | ty::GeneratorWitness(_)
            | ty::Opaque(..) => {
                if debug {
                    output.push_str(&format!("`{:?}`", t));
                } else {
                    bug!(
                        "DefPathBasedNames: trying to create type name for unexpected type: {:?}",
                        t,
                    );
                }
            }
        }
    }

    // Pushes the the name of the specified const to the provided string.
    // If `debug` is true, usually-unprintable consts (such as `Infer`) will be printed,
    // as well as the unprintable types of constants (see `push_type_name` for more details).
    pub fn push_const_name(&self, c: &Const<'tcx>, output: &mut String, debug: bool) {
        match c.val {
            ConstValue::Scalar(..) | ConstValue::Slice { .. } | ConstValue::ByRef { .. } => {
                // FIXME(const_generics): we could probably do a better job here.
                write!(output, "{:?}", c).unwrap()
            }
            _ => {
                if debug {
                    write!(output, "{:?}", c).unwrap()
                } else {
                    bug!(
                        "DefPathBasedNames: trying to create const name for unexpected const: {:?}",
                        c,
                    );
                }
            }
        }
        output.push_str(": ");
        self.push_type_name(c.ty, output, debug);
    }

    pub fn push_def_path(&self, def_id: DefId, output: &mut String) {
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
                write!(output, "{}[{}]::", part.data.as_interned_str(), part.disambiguator)
                    .unwrap();
            }
        }

        // remove final "::"
        output.pop();
        output.pop();
    }

    fn push_generic_params<I>(
        &self,
        substs: SubstsRef<'tcx>,
        projections: I,
        output: &mut String,
        debug: bool,
    ) where
        I: Iterator<Item = ty::PolyExistentialProjection<'tcx>>,
    {
        let mut projections = projections.peekable();
        if substs.non_erasable_generics().next().is_none() && projections.peek().is_none() {
            return;
        }

        output.push('<');

        for type_parameter in substs.types() {
            self.push_type_name(type_parameter, output, debug);
            output.push_str(", ");
        }

        for projection in projections {
            let projection = projection.skip_binder();
            let name = &self.tcx.associated_item(projection.item_def_id).ident.as_str();
            output.push_str(name);
            output.push_str("=");
            self.push_type_name(projection.ty, output, debug);
            output.push_str(", ");
        }

        for const_parameter in substs.consts() {
            self.push_const_name(const_parameter, output, debug);
            output.push_str(", ");
        }

        output.pop();
        output.pop();

        output.push('>');
    }

    pub fn push_instance_as_string(
        &self,
        instance: Instance<'tcx>,
        output: &mut String,
        debug: bool,
    ) {
        self.push_def_path(instance.def_id(), output);
        self.push_generic_params(instance.substs, iter::empty(), output, debug);
    }
}
