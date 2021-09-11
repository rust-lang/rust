// Type Names for Debug Info.

// Notes on targetting MSVC:
// In general, MSVC's debugger attempts to parse all arguments as C++ expressions,
// even if the argument is explicitly a symbol name.
// As such, there are many things that cause parsing issues:
// * `#` is treated as a special character for macros.
// * `{` or `<` at the beginning of a name is treated as an operator.
// * `>>` is always treated as a right-shift.
// * `[` in a name is treated like a regex bracket expression (match any char
//   within the brackets).
// * `"` is treated as the start of a string.

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::definitions::{DefPathData, DefPathDataName, DisambiguatedDefPathData};
use rustc_middle::ich::NodeIdHashingMode;
use rustc_middle::ty::layout::IntegerExt;
use rustc_middle::ty::subst::{GenericArgKind, SubstsRef};
use rustc_middle::ty::{self, AdtDef, ExistentialProjection, Ty, TyCtxt};
use rustc_target::abi::{Integer, TagEncoding, Variants};
use smallvec::SmallVec;

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
    let _prof = tcx.prof.generic_activity("compute_debuginfo_type_name");

    let mut result = String::with_capacity(64);
    let mut visited = FxHashSet::default();
    push_debuginfo_type_name(tcx, t, qualified, &mut result, &mut visited);
    result
}

// Pushes the name of the type as it should be stored in debuginfo on the
// `output` String. See also compute_debuginfo_type_name().
fn push_debuginfo_type_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    t: Ty<'tcx>,
    qualified: bool,
    output: &mut String,
    visited: &mut FxHashSet<Ty<'tcx>>,
) {
    // When targeting MSVC, emit C++ style type names for compatibility with
    // .natvis visualizers (and perhaps other existing native debuggers?)
    let cpp_like_names = cpp_like_names(tcx);

    match *t.kind() {
        ty::Bool => output.push_str("bool"),
        ty::Char => output.push_str("char"),
        ty::Str => output.push_str("str"),
        ty::Never => {
            if cpp_like_names {
                output.push_str("never$");
            } else {
                output.push('!');
            }
        }
        ty::Int(int_ty) => output.push_str(int_ty.name_str()),
        ty::Uint(uint_ty) => output.push_str(uint_ty.name_str()),
        ty::Float(float_ty) => output.push_str(float_ty.name_str()),
        ty::Foreign(def_id) => push_item_name(tcx, def_id, qualified, output),
        ty::Adt(def, substs) => {
            if def.is_enum() && cpp_like_names {
                msvc_enum_fallback(tcx, t, def, substs, output, visited);
            } else {
                push_item_name(tcx, def.did, qualified, output);
                push_generic_params_internal(tcx, substs, output, visited);
            }
        }
        ty::Tuple(component_types) => {
            if cpp_like_names {
                output.push_str("tuple$<");
            } else {
                output.push('(');
            }

            for component_type in component_types {
                push_debuginfo_type_name(tcx, component_type.expect_ty(), true, output, visited);
                push_arg_separator(cpp_like_names, output);
            }
            if !component_types.is_empty() {
                pop_arg_separator(output);
            }

            if cpp_like_names {
                push_close_angle_bracket(cpp_like_names, output);
            } else {
                output.push(')');
            }
        }
        ty::RawPtr(ty::TypeAndMut { ty: inner_type, mutbl }) => {
            if cpp_like_names {
                match mutbl {
                    hir::Mutability::Not => output.push_str("ptr_const$<"),
                    hir::Mutability::Mut => output.push_str("ptr_mut$<"),
                }
            } else {
                output.push('*');
                match mutbl {
                    hir::Mutability::Not => output.push_str("const "),
                    hir::Mutability::Mut => output.push_str("mut "),
                }
            }

            push_debuginfo_type_name(tcx, inner_type, qualified, output, visited);

            if cpp_like_names {
                push_close_angle_bracket(cpp_like_names, output);
            }
        }
        ty::Ref(_, inner_type, mutbl) => {
            // Slices and `&str` are treated like C++ pointers when computing debug
            // info for MSVC debugger. However, wrapping these types' names in a synthetic type
            // causes the .natvis engine for WinDbg to fail to display their data, so we opt these
            // types out to aid debugging in MSVC.
            let is_slice_or_str = match *inner_type.kind() {
                ty::Slice(_) | ty::Str => true,
                _ => false,
            };

            if !cpp_like_names {
                output.push('&');
                output.push_str(mutbl.prefix_str());
            } else if !is_slice_or_str {
                match mutbl {
                    hir::Mutability::Not => output.push_str("ref$<"),
                    hir::Mutability::Mut => output.push_str("ref_mut$<"),
                }
            }

            push_debuginfo_type_name(tcx, inner_type, qualified, output, visited);

            if cpp_like_names && !is_slice_or_str {
                push_close_angle_bracket(cpp_like_names, output);
            }
        }
        ty::Array(inner_type, len) => {
            if cpp_like_names {
                output.push_str("array$<");
                push_debuginfo_type_name(tcx, inner_type, true, output, visited);
                match len.val {
                    ty::ConstKind::Param(param) => write!(output, ",{}>", param.name).unwrap(),
                    _ => write!(output, ",{}>", len.eval_usize(tcx, ty::ParamEnv::reveal_all()))
                        .unwrap(),
                }
            } else {
                output.push('[');
                push_debuginfo_type_name(tcx, inner_type, true, output, visited);
                match len.val {
                    ty::ConstKind::Param(param) => write!(output, "; {}]", param.name).unwrap(),
                    _ => write!(output, "; {}]", len.eval_usize(tcx, ty::ParamEnv::reveal_all()))
                        .unwrap(),
                }
            }
        }
        ty::Slice(inner_type) => {
            if cpp_like_names {
                output.push_str("slice$<");
            } else {
                output.push('[');
            }

            push_debuginfo_type_name(tcx, inner_type, true, output, visited);

            if cpp_like_names {
                push_close_angle_bracket(cpp_like_names, output);
            } else {
                output.push(']');
            }
        }
        ty::Dynamic(ref trait_data, ..) => {
            let auto_traits: SmallVec<[DefId; 4]> = trait_data.auto_traits().collect();

            let has_enclosing_parens = if cpp_like_names {
                output.push_str("dyn$<");
                false
            } else {
                if trait_data.len() > 1 && auto_traits.len() != 0 {
                    // We need enclosing parens because there is more than one trait
                    output.push_str("(dyn ");
                    true
                } else {
                    output.push_str("dyn ");
                    false
                }
            };

            if let Some(principal) = trait_data.principal() {
                let principal =
                    tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), principal);
                push_item_name(tcx, principal.def_id, qualified, output);
                let principal_has_generic_params =
                    push_generic_params_internal(tcx, principal.substs, output, visited);

                let projection_bounds: SmallVec<[_; 4]> = trait_data
                    .projection_bounds()
                    .map(|bound| {
                        let ExistentialProjection { item_def_id, ty, .. } = bound.skip_binder();
                        (item_def_id, ty)
                    })
                    .collect();

                if projection_bounds.len() != 0 {
                    if principal_has_generic_params {
                        // push_generic_params_internal() above added a `>` but we actually
                        // want to add more items to that list, so remove that again.
                        pop_close_angle_bracket(output);
                    }

                    for (item_def_id, ty) in projection_bounds {
                        push_arg_separator(cpp_like_names, output);

                        if cpp_like_names {
                            output.push_str("assoc$<");
                            push_item_name(tcx, item_def_id, false, output);
                            push_arg_separator(cpp_like_names, output);
                            push_debuginfo_type_name(tcx, ty, true, output, visited);
                            push_close_angle_bracket(cpp_like_names, output);
                        } else {
                            push_item_name(tcx, item_def_id, false, output);
                            output.push('=');
                            push_debuginfo_type_name(tcx, ty, true, output, visited);
                        }
                    }

                    push_close_angle_bracket(cpp_like_names, output);
                }

                if auto_traits.len() != 0 {
                    push_auto_trait_separator(cpp_like_names, output);
                }
            }

            if auto_traits.len() != 0 {
                let mut auto_traits: SmallVec<[String; 4]> = auto_traits
                    .into_iter()
                    .map(|def_id| {
                        let mut name = String::with_capacity(20);
                        push_item_name(tcx, def_id, true, &mut name);
                        name
                    })
                    .collect();
                auto_traits.sort_unstable();

                for auto_trait in auto_traits {
                    output.push_str(&auto_trait);
                    push_auto_trait_separator(cpp_like_names, output);
                }

                pop_auto_trait_separator(output);
            }

            if cpp_like_names {
                push_close_angle_bracket(cpp_like_names, output);
            } else if has_enclosing_parens {
                output.push(')');
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
                output.push_str(if cpp_like_names {
                    "recursive_type$"
                } else {
                    "<recursive_type>"
                });
                return;
            }

            let sig =
                tcx.normalize_erasing_late_bound_regions(ty::ParamEnv::reveal_all(), t.fn_sig(tcx));

            if cpp_like_names {
                // Format as a C++ function pointer: return_type (*)(params...)
                if sig.output().is_unit() {
                    output.push_str("void");
                } else {
                    push_debuginfo_type_name(tcx, sig.output(), true, output, visited);
                }
                output.push_str(" (*)(");
            } else {
                output.push_str(sig.unsafety.prefix_str());

                if sig.abi != rustc_target::spec::abi::Abi::Rust {
                    output.push_str("extern \"");
                    output.push_str(sig.abi.name());
                    output.push_str("\" ");
                }

                output.push_str("fn(");
            }

            if !sig.inputs().is_empty() {
                for &parameter_type in sig.inputs() {
                    push_debuginfo_type_name(tcx, parameter_type, true, output, visited);
                    push_arg_separator(cpp_like_names, output);
                }
                pop_arg_separator(output);
            }

            if sig.c_variadic {
                if !sig.inputs().is_empty() {
                    output.push_str(", ...");
                } else {
                    output.push_str("...");
                }
            }

            output.push(')');

            if !cpp_like_names && !sig.output().is_unit() {
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
        ty::Closure(def_id, ..) | ty::Generator(def_id, ..) => {
            let key = tcx.def_key(def_id);
            if qualified {
                let parent_def_id = DefId { index: key.parent.unwrap(), ..def_id };
                push_item_name(tcx, parent_def_id, true, output);
                output.push_str("::");
            }
            push_unqualified_item_name(tcx, def_id, key.disambiguated_data, output);
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

        output.push_str("enum$<");
        push_item_name(tcx, def.did, true, output);
        push_generic_params_internal(tcx, substs, output, visited);

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
                dataful_variant_layout.largest_niche.unwrap().scalar.valid_range;

            let min = dataful_discriminant_range.start;
            let min = tag.value.size(&tcx).truncate(min);

            let max = dataful_discriminant_range.end;
            let max = tag.value.size(&tcx).truncate(max);

            let dataful_variant_name = def.variants[*dataful_variant].ident.as_str();

            output.push_str(&format!(", {}, {}, {}", min, max, dataful_variant_name));
        } else if let Variants::Single { index: variant_idx } = &layout.variants {
            // Uninhabited enums can't be constructed and should never need to be visualized so
            // skip this step for them.
            if def.variants.len() != 0 {
                let variant = def.variants[*variant_idx].ident.as_str();

                output.push_str(&format!(", {}", variant));
            }
        }
        push_close_angle_bracket(true, output);
    }

    const NON_CPP_AUTO_TRAIT_SEPARATOR: &str = " + ";

    fn push_auto_trait_separator(cpp_like_names: bool, output: &mut String) {
        if cpp_like_names {
            push_arg_separator(cpp_like_names, output);
        } else {
            output.push_str(NON_CPP_AUTO_TRAIT_SEPARATOR);
        }
    }

    fn pop_auto_trait_separator(output: &mut String) {
        if output.ends_with(NON_CPP_AUTO_TRAIT_SEPARATOR) {
            output.truncate(output.len() - NON_CPP_AUTO_TRAIT_SEPARATOR.len());
        } else {
            pop_arg_separator(output);
        }
    }
}

pub fn push_item_name(tcx: TyCtxt<'tcx>, def_id: DefId, qualified: bool, output: &mut String) {
    let def_key = tcx.def_key(def_id);
    if qualified {
        if let Some(parent) = def_key.parent {
            push_item_name(tcx, DefId { krate: def_id.krate, index: parent }, true, output);
            output.push_str("::");
        }
    }

    push_unqualified_item_name(tcx, def_id, def_key.disambiguated_data, output);
}

fn push_unqualified_item_name(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    disambiguated_data: DisambiguatedDefPathData,
    output: &mut String,
) {
    match disambiguated_data.data {
        DefPathData::CrateRoot => {
            output.push_str(&tcx.crate_name(def_id.krate).as_str());
        }
        DefPathData::ClosureExpr if tcx.generator_kind(def_id).is_some() => {
            // Generators look like closures, but we want to treat them differently
            // in the debug info.
            if cpp_like_names(tcx) {
                write!(output, "generator${}", disambiguated_data.disambiguator).unwrap();
            } else {
                write!(output, "{{generator#{}}}", disambiguated_data.disambiguator).unwrap();
            }
        }
        _ => match disambiguated_data.data.name() {
            DefPathDataName::Named(name) => {
                output.push_str(&name.as_str());
            }
            DefPathDataName::Anon { namespace } => {
                if cpp_like_names(tcx) {
                    write!(output, "{}${}", namespace, disambiguated_data.disambiguator).unwrap();
                } else {
                    write!(output, "{{{}#{}}}", namespace, disambiguated_data.disambiguator)
                        .unwrap();
                }
            }
        },
    };
}

// Pushes the generic parameters in the given `InternalSubsts` to the output string.
// This ignores region parameters, since they can't reliably be
// reconstructed for items from non-local crates. For local crates, this
// would be possible but with inlining and LTO we have to use the least
// common denominator - otherwise we would run into conflicts.
fn push_generic_params_internal<'tcx>(
    tcx: TyCtxt<'tcx>,
    substs: SubstsRef<'tcx>,
    output: &mut String,
    visited: &mut FxHashSet<Ty<'tcx>>,
) -> bool {
    if substs.non_erasable_generics().next().is_none() {
        return false;
    }

    debug_assert_eq!(substs, tcx.normalize_erasing_regions(ty::ParamEnv::reveal_all(), substs));

    let cpp_like_names = cpp_like_names(tcx);

    output.push('<');

    for type_parameter in substs.non_erasable_generics() {
        match type_parameter {
            GenericArgKind::Type(type_parameter) => {
                push_debuginfo_type_name(tcx, type_parameter, true, output, visited);
            }
            GenericArgKind::Const(ct) => {
                push_const_param(tcx, ct, output);
            }
            other => bug!("Unexpected non-erasable generic: {:?}", other),
        }

        push_arg_separator(cpp_like_names, output);
    }
    pop_arg_separator(output);
    push_close_angle_bracket(cpp_like_names, output);

    true
}

fn push_const_param<'tcx>(tcx: TyCtxt<'tcx>, ct: &'tcx ty::Const<'tcx>, output: &mut String) {
    match ct.val {
        ty::ConstKind::Param(param) => {
            write!(output, "{}", param.name)
        }
        _ => match ct.ty.kind() {
            ty::Int(ity) => {
                let bits = ct.eval_bits(tcx, ty::ParamEnv::reveal_all(), ct.ty);
                let val = Integer::from_int_ty(&tcx, *ity).size().sign_extend(bits) as i128;
                write!(output, "{}", val)
            }
            ty::Uint(_) => {
                let val = ct.eval_bits(tcx, ty::ParamEnv::reveal_all(), ct.ty);
                write!(output, "{}", val)
            }
            ty::Bool => {
                let val = ct.try_eval_bool(tcx, ty::ParamEnv::reveal_all()).unwrap();
                write!(output, "{}", val)
            }
            _ => {
                // If we cannot evaluate the constant to a known type, we fall back
                // to emitting a stable hash value of the constant. This isn't very pretty
                // but we get a deterministic, virtually unique value for the constant.
                let hcx = &mut tcx.create_stable_hashing_context();
                let mut hasher = StableHasher::new();
                hcx.while_hashing_spans(false, |hcx| {
                    hcx.with_node_id_hashing_mode(NodeIdHashingMode::HashDefPath, |hcx| {
                        ct.val.hash_stable(hcx, &mut hasher);
                    });
                });
                // Let's only emit 64 bits of the hash value. That should be plenty for
                // avoiding collisions and will make the emitted type names shorter.
                let hash: u64 = hasher.finish();

                if cpp_like_names(tcx) {
                    write!(output, "CONST${:x}", hash)
                } else {
                    write!(output, "{{CONST#{:x}}}", hash)
                }
            }
        },
    }
    .unwrap();
}

pub fn push_generic_params<'tcx>(tcx: TyCtxt<'tcx>, substs: SubstsRef<'tcx>, output: &mut String) {
    let _prof = tcx.prof.generic_activity("compute_debuginfo_type_name");
    let mut visited = FxHashSet::default();
    push_generic_params_internal(tcx, substs, output, &mut visited);
}

fn push_close_angle_bracket(cpp_like_names: bool, output: &mut String) {
    // MSVC debugger always treats `>>` as a shift, even when parsing templates,
    // so add a space to avoid confusion.
    if cpp_like_names && output.ends_with('>') {
        output.push(' ')
    };

    output.push('>');
}

fn pop_close_angle_bracket(output: &mut String) {
    assert!(output.ends_with('>'), "'output' does not end with '>': {}", output);
    output.pop();
    if output.ends_with(' ') {
        output.pop();
    }
}

fn push_arg_separator(cpp_like_names: bool, output: &mut String) {
    // Natvis does not always like having spaces between parts of the type name
    // and this causes issues when we need to write a typename in natvis, for example
    // as part of a cast like the `HashMap` visualizer does.
    if cpp_like_names {
        output.push(',');
    } else {
        output.push_str(", ");
    };
}

fn pop_arg_separator(output: &mut String) {
    if output.ends_with(' ') {
        output.pop();
    }

    assert!(output.ends_with(','));

    output.pop();
}

fn cpp_like_names(tcx: TyCtxt<'_>) -> bool {
    tcx.sess.target.is_like_msvc
}
