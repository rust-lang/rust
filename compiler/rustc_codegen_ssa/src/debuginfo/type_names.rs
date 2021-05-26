// Type Names for Debug Info.

use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, subst::SubstsRef, AdtDef, Ty, TyCtxt};
use rustc_target::abi::{TagEncoding, Variants};

use std::fmt::Write;

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
    let cpp_like_names = tcx.sess.target.is_like_msvc;

    match *t.kind() {
        ty::Bool => output.push_str("bool"),
        ty::Char => output.push_str("char"),
        ty::Str => output.push_str("str"),
        ty::Never => output.push('!'),
        ty::Int(int_ty) => output.push_str(int_ty.name_str()),
        ty::Uint(uint_ty) => output.push_str(uint_ty.name_str()),
        ty::Float(float_ty) => output.push_str(float_ty.name_str()),
        ty::Foreign(def_id) => push_item_name(tcx, def_id, qualified, output),
        ty::Adt(def, substs) => {
            if def.is_enum() && cpp_like_names {
                msvc_enum_fallback(tcx, t, def, substs, output, visited);
            } else {
                push_item_name(tcx, def.did, qualified, output);
                push_type_params(tcx, substs, output, visited);
            }
        }
        ty::Tuple(component_types) => {
            if cpp_like_names {
                output.push_str("tuple<");
            } else {
                output.push('(');
            }

            for component_type in component_types {
                push_debuginfo_type_name(tcx, component_type.expect_ty(), true, output, visited);
                output.push_str(", ");
            }
            if !component_types.is_empty() {
                output.pop();
                output.pop();
            }

            if cpp_like_names {
                output.push('>');
            } else {
                output.push(')');
            }
        }
        ty::RawPtr(ty::TypeAndMut { ty: inner_type, mutbl }) => {
            if !cpp_like_names {
                output.push('*');
            }
            match mutbl {
                hir::Mutability::Not => output.push_str("const "),
                hir::Mutability::Mut => output.push_str("mut "),
            }

            push_debuginfo_type_name(tcx, inner_type, true, output, visited);

            if cpp_like_names {
                output.push('*');
            }
        }
        ty::Ref(_, inner_type, mutbl) => {
            if !cpp_like_names {
                output.push('&');
            }
            output.push_str(mutbl.prefix_str());

            push_debuginfo_type_name(tcx, inner_type, true, output, visited);

            if cpp_like_names {
                // Slices and `&str` are treated like C++ pointers when computing debug
                // info for MSVC debugger. However, adding '*' at the end of these types' names
                // causes the .natvis engine for WinDbg to fail to display their data, so we opt these
                // types out to aid debugging in MSVC.
                match *inner_type.kind() {
                    ty::Slice(_) | ty::Str => {}
                    _ => output.push('*'),
                }
            }
        }
        ty::Array(inner_type, len) => {
            output.push('[');
            push_debuginfo_type_name(tcx, inner_type, true, output, visited);
            output.push_str(&format!("; {}", len.eval_usize(tcx, ty::ParamEnv::reveal_all())));
            output.push(']');
        }
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
        }
        ty::Dynamic(ref trait_data, ..) => {
            if let Some(principal) = trait_data.principal() {
                let principal =
                    tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), principal);
                push_item_name(tcx, principal.def_id, false, output);
                push_type_params(tcx, principal.substs, output, visited);
            } else {
                output.push_str("dyn '_");
            }
        }
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
            output.push_str(sig.unsafety().prefix_str());

            let abi = sig.abi();
            if abi != rustc_target::spec::abi::Abi::Rust {
                output.push_str("extern \"");
                output.push_str(abi.name());
                output.push_str("\" ");
            }

            output.push_str("fn(");

            let sig = tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), sig);
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
        }
        ty::Closure(def_id, ..) => {
            output.push_str(&format!(
                "closure-{}",
                tcx.def_key(def_id).disambiguated_data.disambiguator
            ));
        }
        ty::Generator(def_id, ..) => {
            output.push_str(&format!(
                "generator-{}",
                tcx.def_key(def_id).disambiguated_data.disambiguator
            ));
        }
        // Type parameters from polymorphized functions.
        ty::Param(_) => {
            output.push_str(&format!("{:?}", t));
        }
        ty::Error(_)
        | ty::Infer(_)
        | ty::Placeholder(..)
        | ty::Projection(..)
        | ty::Bound(..)
        | ty::Opaque(..)
        | ty::GeneratorWitness(..) => {
            bug!(
                "debuginfo: Trying to create type name for \
                  unexpected type: {:?}",
                t
            );
        }
    }

    /// MSVC names enums differently than other platforms so that the debugging visualization
    // format (natvis) is able to understand enums and render the active variant correctly in the
    // debugger. For more information, look in `src/etc/natvis/intrinsic.natvis` and
    // `EnumMemberDescriptionFactor::create_member_descriptions`.
    fn msvc_enum_fallback(
        tcx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
        def: &AdtDef,
        substs: SubstsRef<'tcx>,
        output: &mut String,
        visited: &mut FxHashSet<Ty<'tcx>>,
    ) {
        let layout = tcx.layout_of(tcx.param_env(def.did).and(ty)).expect("layout error");

        if let Variants::Multiple {
            tag_encoding: TagEncoding::Niche { dataful_variant, .. },
            tag,
            variants,
            ..
        } = &layout.variants
        {
            let dataful_variant_layout = &variants[*dataful_variant];

            // calculate the range of values for the dataful variant
            let dataful_discriminant_range =
                &dataful_variant_layout.largest_niche.as_ref().unwrap().scalar.valid_range;

            let min = dataful_discriminant_range.start();
            let min = tag.value.size(&tcx).truncate(*min);

            let max = dataful_discriminant_range.end();
            let max = tag.value.size(&tcx).truncate(*max);

            output.push_str("enum$<");
            push_item_name(tcx, def.did, true, output);
            push_type_params(tcx, substs, output, visited);

            let dataful_variant_name = def.variants[*dataful_variant].ident.as_str();

            output.push_str(&format!(", {}, {}, {}>", min, max, dataful_variant_name));
        } else {
            output.push_str("enum$<");
            push_item_name(tcx, def.did, true, output);
            push_type_params(tcx, substs, output, visited);
            output.push('>');
        }
    }

    fn push_item_name(tcx: TyCtxt<'tcx>, def_id: DefId, qualified: bool, output: &mut String) {
        if qualified {
            output.push_str(&tcx.crate_name(def_id.krate).as_str());
            for path_element in tcx.def_path(def_id).data {
                write!(output, "::{}", path_element.data).unwrap();
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
