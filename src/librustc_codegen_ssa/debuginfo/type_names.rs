// Type Names for Debug Info.

use rustc::hir::{self, def_id::DefId};
use rustc::ty::{self, Ty, TyCtxt, subst::SubstsRef};
use rustc_data_structures::fx::FxHashSet;

// Compute the name of the type as it should be stored in debuginfo. Does not do
// any caching, i.e., calling the function twice with the same type will also do
// the work twice. The `qualified` parameter only affects the first level of the
// type name, further levels (i.e., type parameters) are always fully qualified.
pub fn compute_debuginfo_type_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    t: Ty<'tcx>,
    qualified: bool,
) -> String {
    let mut result = String::with_capacity(64);
    let mut visited = FxHashSet::default();
    push_debuginfo_type_name(tcx, t, qualified, &mut result, &mut visited);
    result
}

// Pushes the name of the type as it should be stored in debuginfo on the
// `output` String. See also compute_debuginfo_type_name().
pub fn push_debuginfo_type_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    t: Ty<'tcx>,
    qualified: bool,
    output: &mut String,
    visited: &mut FxHashSet<Ty<'tcx>>,
) {
    // When targeting MSVC, emit C++ style type names for compatibility with
    // .natvis visualizers (and perhaps other existing native debuggers?)
    let cpp_like_names = tcx.sess.target.target.options.is_like_msvc;

    match t.sty {
        ty::Bool => output.push_str("bool"),
        ty::Char => output.push_str("char"),
        ty::Str => output.push_str("str"),
        ty::Never => output.push_str("!"),
        ty::Int(int_ty) => output.push_str(int_ty.ty_to_string()),
        ty::Uint(uint_ty) => output.push_str(uint_ty.ty_to_string()),
        ty::Float(float_ty) => output.push_str(float_ty.ty_to_string()),
        ty::Foreign(def_id) => push_item_name(tcx, def_id, qualified, output),
        ty::Adt(def, substs) => {
            push_item_name(tcx, def.did, qualified, output);
            push_type_params(tcx, substs, output, visited);
        },
        ty::Tuple(component_types) => {
            output.push('(');
            for &component_type in component_types {
                push_debuginfo_type_name(tcx, component_type.expect_ty(), true, output, visited);
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

            push_debuginfo_type_name(tcx, inner_type, true, output, visited);

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

            push_debuginfo_type_name(tcx, inner_type, true, output, visited);

            if cpp_like_names {
                output.push('*');
            }
        },
        ty::Array(inner_type, len) => {
            output.push('[');
            push_debuginfo_type_name(tcx, inner_type, true, output, visited);
            output.push_str(&format!("; {}", len.unwrap_usize(tcx)));
            output.push(']');
        },
        ty::Slice(inner_type) => {
            if cpp_like_names {
                output.push_str("slice<");
            } else {
                output.push('[');
            }

            push_debuginfo_type_name(tcx, inner_type, true, output, visited);

            if cpp_like_names {
                output.push('>');
            } else {
                output.push(']');
            }
        },
        ty::Dynamic(ref trait_data, ..) => {
            if let Some(principal) = trait_data.principal() {
                let principal = tcx.normalize_erasing_late_bound_regions(
                    ty::ParamEnv::reveal_all(),
                    &principal,
                );
                push_item_name(tcx, principal.def_id, false, output);
                push_type_params(tcx, principal.substs, output, visited);
            } else {
                output.push_str("dyn '_");
            }
        },
        ty::FnDef(..) | ty::FnPtr(_) => {
            // We've encountered a weird 'recursive type'
            // Currently, the only way to generate such a type
            // is by using 'impl trait':
            //
            // fn foo() -> impl Copy { foo }
            //
            // There's not really a sensible name we can generate,
            // since we don't include 'impl trait' types (e.g. ty::Opaque)
            // in the output
            //
            // Since we need to generate *something*, we just
            // use a dummy string that should make it clear
            // that something unusual is going on
            if !visited.insert(t) {
                output.push_str("<recursive_type>");
                return;
            }


            let sig = t.fn_sig(tcx);
            if sig.unsafety() == hir::Unsafety::Unsafe {
                output.push_str("unsafe ");
            }

            let abi = sig.abi();
            if abi != rustc_target::spec::abi::Abi::Rust {
                output.push_str("extern \"");
                output.push_str(abi.name());
                output.push_str("\" ");
            }

            output.push_str("fn(");

            let sig = tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), &sig);
            if !sig.inputs().is_empty() {
                for &parameter_type in sig.inputs() {
                    push_debuginfo_type_name(tcx, parameter_type, true, output, visited);
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
                push_debuginfo_type_name(tcx, sig.output(), true, output, visited);
            }


            // We only keep the type in 'visited'
            // for the duration of the body of this method.
            // It's fine for a particular function type
            // to show up multiple times in one overall type
            // (e.g. MyType<fn() -> u8, fn() -> u8>
            //
            // We only care about avoiding recursing
            // directly back to the type we're currently
            // processing
            visited.remove(t);
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

    fn push_item_name(tcx: TyCtxt<'tcx>, def_id: DefId, qualified: bool, output: &mut String) {
        if qualified {
            output.push_str(&tcx.crate_name(def_id.krate).as_str());
            for path_element in tcx.def_path(def_id).data {
                output.push_str("::");
                output.push_str(&path_element.data.as_interned_str().as_str());
            }
        } else {
            output.push_str(&tcx.item_name(def_id).as_str());
        }
    }

    // Pushes the type parameters in the given `InternalSubsts` to the output string.
    // This ignores region parameters, since they can't reliably be
    // reconstructed for items from non-local crates. For local crates, this
    // would be possible but with inlining and LTO we have to use the least
    // common denominator - otherwise we would run into conflicts.
    fn push_type_params<'tcx>(
        tcx: TyCtxt<'tcx>,
        substs: SubstsRef<'tcx>,
        output: &mut String,
        visited: &mut FxHashSet<Ty<'tcx>>,
    ) {
        if substs.types().next().is_none() {
            return;
        }

        output.push('<');

        for type_parameter in substs.types() {
            push_debuginfo_type_name(tcx, type_parameter, true, output, visited);
            output.push_str(", ");
        }

        output.pop();
        output.pop();

        output.push('>');
    }
}
