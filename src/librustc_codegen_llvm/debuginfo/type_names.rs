// Type Names for Debug Info.

use common::CodegenCx;
use rustc::hir::def_id::DefId;
use rustc::ty::subst::Substs;
use rustc::ty::{self, Ty};
use rustc_codegen_ssa::traits::*;

use rustc::hir;

// Compute the name of the type as it should be stored in debuginfo. Does not do
// any caching, i.e., calling the function twice with the same type will also do
// the work twice. The `qualified` parameter only affects the first level of the
// type name, further levels (i.e., type parameters) are always fully qualified.
pub fn compute_debuginfo_type_name<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                             t: Ty<'tcx>,
                                             qualified: bool)
                                             -> String {
    let mut result = String::with_capacity(64);
    push_debuginfo_type_name(cx, t, qualified, &mut result);
    result
}

// Pushes the name of the type as it should be stored in debuginfo on the
// `output` String. See also compute_debuginfo_type_name().
pub fn push_debuginfo_type_name<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                          t: Ty<'tcx>,
                                          qualified: bool,
                                          output: &mut String) {
    // When targeting MSVC, emit C++ style type names for compatibility with
    // .natvis visualizers (and perhaps other existing native debuggers?)
    let cpp_like_names = cx.sess().target.target.options.is_like_msvc;

    match t.sty {
        ty::Bool => output.push_str("bool"),
        ty::Char => output.push_str("char"),
        ty::Str => output.push_str("str"),
        ty::Never => output.push_str("!"),
        ty::Int(int_ty) => output.push_str(int_ty.ty_to_string()),
        ty::Uint(uint_ty) => output.push_str(uint_ty.ty_to_string()),
        ty::Float(float_ty) => output.push_str(float_ty.ty_to_string()),
        ty::Foreign(def_id) => push_item_name(cx, def_id, qualified, output),
        ty::Adt(def, substs) => {
            push_item_name(cx, def.did, qualified, output);
            push_type_params(cx, substs, output);
        },
        ty::Tuple(component_types) => {
            output.push('(');
            for &component_type in component_types {
                push_debuginfo_type_name(cx, component_type, true, output);
                output.push_str(", ");
            }
            if !component_types.is_empty() {
                output.pop();
                output.pop();
            }
            output.push(')');
        },
        ty::RawPtr(ty::TypeAndMut { ty: inner_type, mutbl } ) => {
            if !cpp_like_names {
                output.push('*');
            }
            match mutbl {
                hir::MutImmutable => output.push_str("const "),
                hir::MutMutable => output.push_str("mut "),
            }

            push_debuginfo_type_name(cx, inner_type, true, output);

            if cpp_like_names {
                output.push('*');
            }
        },
        ty::Ref(_, inner_type, mutbl) => {
            if !cpp_like_names {
                output.push('&');
            }
            if mutbl == hir::MutMutable {
                output.push_str("mut ");
            }

            push_debuginfo_type_name(cx, inner_type, true, output);

            if cpp_like_names {
                output.push('*');
            }
        },
        ty::Array(inner_type, len) => {
            output.push('[');
            push_debuginfo_type_name(cx, inner_type, true, output);
            output.push_str(&format!("; {}", len.unwrap_usize(cx.tcx)));
            output.push(']');
        },
        ty::Slice(inner_type) => {
            if cpp_like_names {
                output.push_str("slice<");
            } else {
                output.push('[');
            }

            push_debuginfo_type_name(cx, inner_type, true, output);

            if cpp_like_names {
                output.push('>');
            } else {
                output.push(']');
            }
        },
        ty::Dynamic(ref trait_data, ..) => {
            let principal = cx.tcx.normalize_erasing_late_bound_regions(
                ty::ParamEnv::reveal_all(),
                &trait_data.principal(),
            );
            push_item_name(cx, principal.def_id, false, output);
            push_type_params(cx, principal.substs, output);
        },
        ty::FnDef(..) | ty::FnPtr(_) => {
            let sig = t.fn_sig(cx.tcx);
            if sig.unsafety() == hir::Unsafety::Unsafe {
                output.push_str("unsafe ");
            }

            let abi = sig.abi();
            if abi != ::abi::Abi::Rust {
                output.push_str("extern \"");
                output.push_str(abi.name());
                output.push_str("\" ");
            }

            output.push_str("fn(");

            let sig = cx.tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);
            if !sig.inputs().is_empty() {
                for &parameter_type in sig.inputs() {
                    push_debuginfo_type_name(cx, parameter_type, true, output);
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
                push_debuginfo_type_name(cx, sig.output(), true, output);
            }
        },
        ty::Closure(..) => {
            output.push_str("closure");
        }
        ty::Generator(..) => {
            output.push_str("generator");
        }
        ty::Error |
        ty::Infer(_) |
        ty::Placeholder(..) |
        ty::UnnormalizedProjection(..) |
        ty::Projection(..) |
        ty::Bound(..) |
        ty::Opaque(..) |
        ty::GeneratorWitness(..) |
        ty::Param(_) => {
            bug!("debuginfo: Trying to create type name for \
                  unexpected type: {:?}", t);
        }
    }

    fn push_item_name(cx: &CodegenCx,
                      def_id: DefId,
                      qualified: bool,
                      output: &mut String) {
        if qualified {
            output.push_str(&cx.tcx.crate_name(def_id.krate).as_str());
            for path_element in cx.tcx.def_path(def_id).data {
                output.push_str("::");
                output.push_str(&path_element.data.as_interned_str().as_str());
            }
        } else {
            output.push_str(&cx.tcx.item_name(def_id).as_str());
        }
    }

    // Pushes the type parameters in the given `Substs` to the output string.
    // This ignores region parameters, since they can't reliably be
    // reconstructed for items from non-local crates. For local crates, this
    // would be possible but with inlining and LTO we have to use the least
    // common denominator - otherwise we would run into conflicts.
    fn push_type_params<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                  substs: &Substs<'tcx>,
                                  output: &mut String) {
        if substs.types().next().is_none() {
            return;
        }

        output.push('<');

        for type_parameter in substs.types() {
            push_debuginfo_type_name(cx, type_parameter, true, output);
            output.push_str(", ");
        }

        output.pop();
        output.pop();

        output.push('>');
    }
}
