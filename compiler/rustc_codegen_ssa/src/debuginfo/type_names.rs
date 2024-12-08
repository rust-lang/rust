//! Type Names for Debug Info.

// Notes on targeting MSVC:
// In general, MSVC's debugger attempts to parse all arguments as C++ expressions,
// even if the argument is explicitly a symbol name.
// As such, there are many things that cause parsing issues:
// * `#` is treated as a special character for macros.
// * `{` or `<` at the beginning of a name is treated as an operator.
// * `>>` is always treated as a right-shift.
// * `[` in a name is treated like a regex bracket expression (match any char
//   within the brackets).
// * `"` is treated as the start of a string.

use std::fmt::Write;

use rustc_abi::Integer;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::stable_hasher::{Hash64, HashStable, StableHasher};
use rustc_hir::def_id::DefId;
use rustc_hir::definitions::{DefPathData, DefPathDataName, DisambiguatedDefPathData};
use rustc_hir::{CoroutineDesugaring, CoroutineKind, CoroutineSource, Mutability};
use rustc_middle::bug;
use rustc_middle::ty::layout::{IntegerExt, TyAndLayout};
use rustc_middle::ty::{self, ExistentialProjection, GenericArgKind, GenericArgsRef, Ty, TyCtxt};
use smallvec::SmallVec;

use crate::debuginfo::wants_c_like_enum_debuginfo;

/// Compute the name of the type as it should be stored in debuginfo. Does not do
/// any caching, i.e., calling the function twice with the same type will also do
/// the work twice. The `qualified` parameter only affects the first level of the
/// type name, further levels (i.e., type parameters) are always fully qualified.
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
    let cpp_like_debuginfo = cpp_like_debuginfo(tcx);

    match *t.kind() {
        ty::Bool => output.push_str("bool"),
        ty::Char => output.push_str("char"),
        ty::Str => {
            if cpp_like_debuginfo {
                output.push_str("str$")
            } else {
                output.push_str("str")
            }
        }
        ty::Never => {
            if cpp_like_debuginfo {
                output.push_str("never$");
            } else {
                output.push('!');
            }
        }
        ty::Int(int_ty) => output.push_str(int_ty.name_str()),
        ty::Uint(uint_ty) => output.push_str(uint_ty.name_str()),
        ty::Float(float_ty) => output.push_str(float_ty.name_str()),
        ty::Foreign(def_id) => push_item_name(tcx, def_id, qualified, output),
        ty::Adt(def, args) => {
            // `layout_for_cpp_like_fallback` will be `Some` if we want to use the fallback encoding.
            let layout_for_cpp_like_fallback = if cpp_like_debuginfo && def.is_enum() {
                match tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(t)) {
                    Ok(layout) => {
                        if !wants_c_like_enum_debuginfo(tcx, layout) {
                            Some(layout)
                        } else {
                            // This is a C-like enum so we don't want to use the fallback encoding
                            // for the name.
                            None
                        }
                    }
                    Err(e) => {
                        // Computing the layout can still fail here, e.g. if the target architecture
                        // cannot represent the type. See
                        // https://github.com/rust-lang/rust/issues/94961.
                        tcx.dcx().emit_fatal(e.into_diagnostic());
                    }
                }
            } else {
                // We are not emitting cpp-like debuginfo or this isn't even an enum.
                None
            };

            if let Some(ty_and_layout) = layout_for_cpp_like_fallback {
                msvc_enum_fallback(
                    tcx,
                    ty_and_layout,
                    &|output, visited| {
                        push_item_name(tcx, def.did(), true, output);
                        push_generic_params_internal(tcx, args, output, visited);
                    },
                    output,
                    visited,
                );
            } else {
                push_item_name(tcx, def.did(), qualified, output);
                push_generic_params_internal(tcx, args, output, visited);
            }
        }
        ty::Tuple(component_types) => {
            if cpp_like_debuginfo {
                output.push_str("tuple$<");
            } else {
                output.push('(');
            }

            for component_type in component_types {
                push_debuginfo_type_name(tcx, component_type, true, output, visited);
                push_arg_separator(cpp_like_debuginfo, output);
            }
            if !component_types.is_empty() {
                pop_arg_separator(output);
            }

            if cpp_like_debuginfo {
                push_close_angle_bracket(cpp_like_debuginfo, output);
            } else {
                output.push(')');
            }
        }
        ty::RawPtr(inner_type, mutbl) => {
            if cpp_like_debuginfo {
                match mutbl {
                    Mutability::Not => output.push_str("ptr_const$<"),
                    Mutability::Mut => output.push_str("ptr_mut$<"),
                }
            } else {
                output.push('*');
                match mutbl {
                    Mutability::Not => output.push_str("const "),
                    Mutability::Mut => output.push_str("mut "),
                }
            }

            push_debuginfo_type_name(tcx, inner_type, qualified, output, visited);

            if cpp_like_debuginfo {
                push_close_angle_bracket(cpp_like_debuginfo, output);
            }
        }
        ty::Ref(_, inner_type, mutbl) => {
            if cpp_like_debuginfo {
                match mutbl {
                    Mutability::Not => output.push_str("ref$<"),
                    Mutability::Mut => output.push_str("ref_mut$<"),
                }
            } else {
                output.push('&');
                output.push_str(mutbl.prefix_str());
            }

            push_debuginfo_type_name(tcx, inner_type, qualified, output, visited);

            if cpp_like_debuginfo {
                push_close_angle_bracket(cpp_like_debuginfo, output);
            }
        }
        ty::Array(inner_type, len) => {
            if cpp_like_debuginfo {
                output.push_str("array$<");
                push_debuginfo_type_name(tcx, inner_type, true, output, visited);
                match len.kind() {
                    ty::ConstKind::Param(param) => write!(output, ",{}>", param.name).unwrap(),
                    _ => write!(
                        output,
                        ",{}>",
                        len.try_to_target_usize(tcx)
                            .expect("expected monomorphic const in codegen")
                    )
                    .unwrap(),
                }
            } else {
                output.push('[');
                push_debuginfo_type_name(tcx, inner_type, true, output, visited);
                match len.kind() {
                    ty::ConstKind::Param(param) => write!(output, "; {}]", param.name).unwrap(),
                    _ => write!(
                        output,
                        "; {}]",
                        len.try_to_target_usize(tcx)
                            .expect("expected monomorphic const in codegen")
                    )
                    .unwrap(),
                }
            }
        }
        ty::Pat(inner_type, pat) => {
            if cpp_like_debuginfo {
                output.push_str("pat$<");
                push_debuginfo_type_name(tcx, inner_type, true, output, visited);
                // FIXME(wg-debugging): implement CPP like printing for patterns.
                write!(output, ",{:?}>", pat).unwrap();
            } else {
                write!(output, "{:?}", t).unwrap();
            }
        }
        ty::Slice(inner_type) => {
            if cpp_like_debuginfo {
                output.push_str("slice2$<");
            } else {
                output.push('[');
            }

            push_debuginfo_type_name(tcx, inner_type, true, output, visited);

            if cpp_like_debuginfo {
                push_close_angle_bracket(cpp_like_debuginfo, output);
            } else {
                output.push(']');
            }
        }
        ty::Dynamic(trait_data, ..) => {
            let auto_traits: SmallVec<[DefId; 4]> = trait_data.auto_traits().collect();

            let has_enclosing_parens = if cpp_like_debuginfo {
                output.push_str("dyn$<");
                false
            } else if trait_data.len() > 1 && auto_traits.len() != 0 {
                // We need enclosing parens because there is more than one trait
                output.push_str("(dyn ");
                true
            } else {
                output.push_str("dyn ");
                false
            };

            if let Some(principal) = trait_data.principal() {
                let principal = tcx.normalize_erasing_late_bound_regions(
                    ty::TypingEnv::fully_monomorphized(),
                    principal,
                );
                push_item_name(tcx, principal.def_id, qualified, output);
                let principal_has_generic_params =
                    push_generic_params_internal(tcx, principal.args, output, visited);

                let projection_bounds: SmallVec<[_; 4]> = trait_data
                    .projection_bounds()
                    .map(|bound| {
                        let ExistentialProjection { def_id: item_def_id, term, .. } =
                            tcx.instantiate_bound_regions_with_erased(bound);
                        // FIXME(associated_const_equality): allow for consts here
                        (item_def_id, term.expect_type())
                    })
                    .collect();

                if projection_bounds.len() != 0 {
                    if principal_has_generic_params {
                        // push_generic_params_internal() above added a `>` but we actually
                        // want to add more items to that list, so remove that again...
                        pop_close_angle_bracket(output);
                        // .. and add a comma to separate the regular generic args from the
                        // associated types.
                        push_arg_separator(cpp_like_debuginfo, output);
                    } else {
                        // push_generic_params_internal() did not add `<...>`, so we open
                        // angle brackets here.
                        output.push('<');
                    }

                    for (item_def_id, ty) in projection_bounds {
                        if cpp_like_debuginfo {
                            output.push_str("assoc$<");
                            push_item_name(tcx, item_def_id, false, output);
                            push_arg_separator(cpp_like_debuginfo, output);
                            push_debuginfo_type_name(tcx, ty, true, output, visited);
                            push_close_angle_bracket(cpp_like_debuginfo, output);
                        } else {
                            push_item_name(tcx, item_def_id, false, output);
                            output.push('=');
                            push_debuginfo_type_name(tcx, ty, true, output, visited);
                        }
                        push_arg_separator(cpp_like_debuginfo, output);
                    }

                    pop_arg_separator(output);
                    push_close_angle_bracket(cpp_like_debuginfo, output);
                }

                if auto_traits.len() != 0 {
                    push_auto_trait_separator(cpp_like_debuginfo, output);
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
                    push_auto_trait_separator(cpp_like_debuginfo, output);
                }

                pop_auto_trait_separator(output);
            }

            if cpp_like_debuginfo {
                push_close_angle_bracket(cpp_like_debuginfo, output);
            } else if has_enclosing_parens {
                output.push(')');
            }
        }
        ty::FnDef(..) | ty::FnPtr(..) => {
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
                output.push_str(if cpp_like_debuginfo {
                    "recursive_type$"
                } else {
                    "<recursive_type>"
                });
                return;
            }

            let sig = tcx.normalize_erasing_late_bound_regions(
                ty::TypingEnv::fully_monomorphized(),
                t.fn_sig(tcx),
            );

            if cpp_like_debuginfo {
                // Format as a C++ function pointer: return_type (*)(params...)
                if sig.output().is_unit() {
                    output.push_str("void");
                } else {
                    push_debuginfo_type_name(tcx, sig.output(), true, output, visited);
                }
                output.push_str(" (*)(");
            } else {
                output.push_str(sig.safety.prefix_str());

                if sig.abi != rustc_abi::ExternAbi::Rust {
                    output.push_str("extern \"");
                    output.push_str(sig.abi.name());
                    output.push_str("\" ");
                }

                output.push_str("fn(");
            }

            if !sig.inputs().is_empty() {
                for &parameter_type in sig.inputs() {
                    push_debuginfo_type_name(tcx, parameter_type, true, output, visited);
                    push_arg_separator(cpp_like_debuginfo, output);
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

            if !cpp_like_debuginfo && !sig.output().is_unit() {
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
            visited.remove(&t);
        }
        ty::Closure(def_id, args)
        | ty::CoroutineClosure(def_id, args)
        | ty::Coroutine(def_id, args, ..) => {
            // Name will be "{closure_env#0}<T1, T2, ...>", "{coroutine_env#0}<T1, T2, ...>", or
            // "{async_fn_env#0}<T1, T2, ...>", etc.
            // In the case of cpp-like debuginfo, the name additionally gets wrapped inside of
            // an artificial `enum2$<>` type, as defined in msvc_enum_fallback().
            if cpp_like_debuginfo && t.is_coroutine() {
                let ty_and_layout =
                    tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(t)).unwrap();
                msvc_enum_fallback(
                    tcx,
                    ty_and_layout,
                    &|output, visited| {
                        push_closure_or_coroutine_name(tcx, def_id, args, true, output, visited);
                    },
                    output,
                    visited,
                );
            } else {
                push_closure_or_coroutine_name(tcx, def_id, args, qualified, output, visited);
            }
        }
        ty::Param(_)
        | ty::Error(_)
        | ty::Infer(_)
        | ty::Placeholder(..)
        | ty::Alias(..)
        | ty::Bound(..)
        | ty::CoroutineWitness(..) => {
            bug!(
                "debuginfo: Trying to create type name for \
                  unexpected type: {:?}",
                t
            );
        }
    }

    /// MSVC names enums differently than other platforms so that the debugging visualization
    // format (natvis) is able to understand enums and render the active variant correctly in the
    // debugger. For more information, look in
    // rustc_codegen_llvm/src/debuginfo/metadata/enums/cpp_like.rs.
    fn msvc_enum_fallback<'tcx>(
        tcx: TyCtxt<'tcx>,
        ty_and_layout: TyAndLayout<'tcx>,
        push_inner: &dyn Fn(/*output*/ &mut String, /*visited*/ &mut FxHashSet<Ty<'tcx>>),
        output: &mut String,
        visited: &mut FxHashSet<Ty<'tcx>>,
    ) {
        assert!(!wants_c_like_enum_debuginfo(tcx, ty_and_layout));
        output.push_str("enum2$<");
        push_inner(output, visited);
        push_close_angle_bracket(true, output);
    }

    const NON_CPP_AUTO_TRAIT_SEPARATOR: &str = " + ";

    fn push_auto_trait_separator(cpp_like_debuginfo: bool, output: &mut String) {
        if cpp_like_debuginfo {
            push_arg_separator(cpp_like_debuginfo, output);
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

pub enum VTableNameKind {
    // Is the name for the const/static holding the vtable?
    GlobalVariable,
    // Is the name for the type of the vtable?
    Type,
}

/// Computes a name for the global variable storing a vtable (or the type of that global variable).
///
/// The name is of the form:
///
/// `<path::to::SomeType as path::to::SomeTrait>::{vtable}`
///
/// or, when generating C++-like names:
///
/// `impl$<path::to::SomeType, path::to::SomeTrait>::vtable$`
///
/// If `kind` is `VTableNameKind::Type` then the last component is `{vtable_ty}` instead of just
/// `{vtable}`, so that the type and the corresponding global variable get assigned different
/// names.
pub fn compute_debuginfo_vtable_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    t: Ty<'tcx>,
    trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
    kind: VTableNameKind,
) -> String {
    let cpp_like_debuginfo = cpp_like_debuginfo(tcx);

    let mut vtable_name = String::with_capacity(64);

    if cpp_like_debuginfo {
        vtable_name.push_str("impl$<");
    } else {
        vtable_name.push('<');
    }

    let mut visited = FxHashSet::default();
    push_debuginfo_type_name(tcx, t, true, &mut vtable_name, &mut visited);

    if cpp_like_debuginfo {
        vtable_name.push_str(", ");
    } else {
        vtable_name.push_str(" as ");
    }

    if let Some(trait_ref) = trait_ref {
        let trait_ref = tcx
            .normalize_erasing_late_bound_regions(ty::TypingEnv::fully_monomorphized(), trait_ref);
        push_item_name(tcx, trait_ref.def_id, true, &mut vtable_name);
        visited.clear();
        push_generic_params_internal(tcx, trait_ref.args, &mut vtable_name, &mut visited);
    } else {
        vtable_name.push('_');
    }

    push_close_angle_bracket(cpp_like_debuginfo, &mut vtable_name);

    let suffix = match (cpp_like_debuginfo, kind) {
        (true, VTableNameKind::GlobalVariable) => "::vtable$",
        (false, VTableNameKind::GlobalVariable) => "::{vtable}",
        (true, VTableNameKind::Type) => "::vtable_type$",
        (false, VTableNameKind::Type) => "::{vtable_type}",
    };

    vtable_name.reserve_exact(suffix.len());
    vtable_name.push_str(suffix);

    vtable_name
}

pub fn push_item_name(tcx: TyCtxt<'_>, def_id: DefId, qualified: bool, output: &mut String) {
    let def_key = tcx.def_key(def_id);
    if qualified {
        if let Some(parent) = def_key.parent {
            push_item_name(tcx, DefId { krate: def_id.krate, index: parent }, true, output);
            output.push_str("::");
        }
    }

    push_unqualified_item_name(tcx, def_id, def_key.disambiguated_data, output);
}

fn coroutine_kind_label(coroutine_kind: Option<CoroutineKind>) -> &'static str {
    use CoroutineDesugaring::*;
    use CoroutineKind::*;
    use CoroutineSource::*;
    match coroutine_kind {
        Some(Desugared(Gen, Block)) => "gen_block",
        Some(Desugared(Gen, Closure)) => "gen_closure",
        Some(Desugared(Gen, Fn)) => "gen_fn",
        Some(Desugared(Async, Block)) => "async_block",
        Some(Desugared(Async, Closure)) => "async_closure",
        Some(Desugared(Async, Fn)) => "async_fn",
        Some(Desugared(AsyncGen, Block)) => "async_gen_block",
        Some(Desugared(AsyncGen, Closure)) => "async_gen_closure",
        Some(Desugared(AsyncGen, Fn)) => "async_gen_fn",
        Some(Coroutine(_)) => "coroutine",
        None => "closure",
    }
}

fn push_disambiguated_special_name(
    label: &str,
    disambiguator: u32,
    cpp_like_debuginfo: bool,
    output: &mut String,
) {
    if cpp_like_debuginfo {
        write!(output, "{label}${disambiguator}").unwrap();
    } else {
        write!(output, "{{{label}#{disambiguator}}}").unwrap();
    }
}

fn push_unqualified_item_name(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    disambiguated_data: DisambiguatedDefPathData,
    output: &mut String,
) {
    match disambiguated_data.data {
        DefPathData::CrateRoot => {
            output.push_str(tcx.crate_name(def_id.krate).as_str());
        }
        DefPathData::Closure => {
            let label = coroutine_kind_label(tcx.coroutine_kind(def_id));

            push_disambiguated_special_name(
                label,
                disambiguated_data.disambiguator,
                cpp_like_debuginfo(tcx),
                output,
            );
        }
        _ => match disambiguated_data.data.name() {
            DefPathDataName::Named(name) => {
                output.push_str(name.as_str());
            }
            DefPathDataName::Anon { namespace } => {
                push_disambiguated_special_name(
                    namespace.as_str(),
                    disambiguated_data.disambiguator,
                    cpp_like_debuginfo(tcx),
                    output,
                );
            }
        },
    };
}

fn push_generic_params_internal<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: GenericArgsRef<'tcx>,
    output: &mut String,
    visited: &mut FxHashSet<Ty<'tcx>>,
) -> bool {
    assert_eq!(args, tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), args));
    let mut args = args.non_erasable_generics().peekable();
    if args.peek().is_none() {
        return false;
    }
    let cpp_like_debuginfo = cpp_like_debuginfo(tcx);

    output.push('<');

    for type_parameter in args {
        match type_parameter {
            GenericArgKind::Type(type_parameter) => {
                push_debuginfo_type_name(tcx, type_parameter, true, output, visited);
            }
            GenericArgKind::Const(ct) => {
                push_const_param(tcx, ct, output);
            }
            other => bug!("Unexpected non-erasable generic: {:?}", other),
        }

        push_arg_separator(cpp_like_debuginfo, output);
    }
    pop_arg_separator(output);
    push_close_angle_bracket(cpp_like_debuginfo, output);

    true
}

fn push_const_param<'tcx>(tcx: TyCtxt<'tcx>, ct: ty::Const<'tcx>, output: &mut String) {
    match ct.kind() {
        ty::ConstKind::Param(param) => {
            write!(output, "{}", param.name)
        }
        ty::ConstKind::Value(ty, valtree) => {
            match ty.kind() {
                ty::Int(ity) => {
                    // FIXME: directly extract the bits from a valtree instead of evaluating an
                    // already evaluated `Const` in order to get the bits.
                    let bits = ct
                        .try_to_bits(tcx, ty::TypingEnv::fully_monomorphized())
                        .expect("expected monomorphic const in codegen");
                    let val = Integer::from_int_ty(&tcx, *ity).size().sign_extend(bits) as i128;
                    write!(output, "{val}")
                }
                ty::Uint(_) => {
                    let val = ct
                        .try_to_bits(tcx, ty::TypingEnv::fully_monomorphized())
                        .expect("expected monomorphic const in codegen");
                    write!(output, "{val}")
                }
                ty::Bool => {
                    let val = ct.try_to_bool().expect("expected monomorphic const in codegen");
                    write!(output, "{val}")
                }
                _ => {
                    // If we cannot evaluate the constant to a known type, we fall back
                    // to emitting a stable hash value of the constant. This isn't very pretty
                    // but we get a deterministic, virtually unique value for the constant.
                    //
                    // Let's only emit 64 bits of the hash value. That should be plenty for
                    // avoiding collisions and will make the emitted type names shorter.
                    let hash_short = tcx.with_stable_hashing_context(|mut hcx| {
                        let mut hasher = StableHasher::new();
                        hcx.while_hashing_spans(false, |hcx| {
                            (ty, valtree).hash_stable(hcx, &mut hasher)
                        });
                        hasher.finish::<Hash64>()
                    });

                    if cpp_like_debuginfo(tcx) {
                        write!(output, "CONST${hash_short:x}")
                    } else {
                        write!(output, "{{CONST#{hash_short:x}}}")
                    }
                }
            }
        }
        _ => bug!("Invalid `Const` during codegen: {:?}", ct),
    }
    .unwrap();
}

pub fn push_generic_params<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: GenericArgsRef<'tcx>,
    output: &mut String,
) {
    let _prof = tcx.prof.generic_activity("compute_debuginfo_type_name");
    let mut visited = FxHashSet::default();
    push_generic_params_internal(tcx, args, output, &mut visited);
}

fn push_closure_or_coroutine_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    args: GenericArgsRef<'tcx>,
    qualified: bool,
    output: &mut String,
    visited: &mut FxHashSet<Ty<'tcx>>,
) {
    // Name will be "{closure_env#0}<T1, T2, ...>", "{coroutine_env#0}<T1, T2, ...>", or
    // "{async_fn_env#0}<T1, T2, ...>", etc.
    let def_key = tcx.def_key(def_id);
    let coroutine_kind = tcx.coroutine_kind(def_id);

    if qualified {
        let parent_def_id = DefId { index: def_key.parent.unwrap(), ..def_id };
        push_item_name(tcx, parent_def_id, true, output);
        output.push_str("::");
    }

    let mut label = String::with_capacity(20);
    write!(&mut label, "{}_env", coroutine_kind_label(coroutine_kind)).unwrap();

    push_disambiguated_special_name(
        &label,
        def_key.disambiguated_data.disambiguator,
        cpp_like_debuginfo(tcx),
        output,
    );

    // We also need to add the generic arguments of the async fn/coroutine or
    // the enclosing function (for closures or async blocks), so that we end
    // up with a unique name for every instantiation.

    // Find the generics of the enclosing function, as defined in the source code.
    let enclosing_fn_def_id = tcx.typeck_root_def_id(def_id);
    let generics = tcx.generics_of(enclosing_fn_def_id);

    // Truncate the args to the length of the above generics. This will cut off
    // anything closure- or coroutine-specific.
    // FIXME(async_closures): This is probably not going to be correct w.r.t.
    // multiple coroutine flavors. Maybe truncate to (parent + 1)?
    let args = args.truncate_to(tcx, generics);
    push_generic_params_internal(tcx, args, output, visited);
}

fn push_close_angle_bracket(cpp_like_debuginfo: bool, output: &mut String) {
    // MSVC debugger always treats `>>` as a shift, even when parsing templates,
    // so add a space to avoid confusion.
    if cpp_like_debuginfo && output.ends_with('>') {
        output.push(' ')
    };

    output.push('>');
}

fn pop_close_angle_bracket(output: &mut String) {
    assert!(output.ends_with('>'), "'output' does not end with '>': {output}");
    output.pop();
    if output.ends_with(' ') {
        output.pop();
    }
}

fn push_arg_separator(cpp_like_debuginfo: bool, output: &mut String) {
    // Natvis does not always like having spaces between parts of the type name
    // and this causes issues when we need to write a typename in natvis, for example
    // as part of a cast like the `HashMap` visualizer does.
    if cpp_like_debuginfo {
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

/// Check if we should generate C++ like names and debug information.
pub fn cpp_like_debuginfo(tcx: TyCtxt<'_>) -> bool {
    tcx.sess.target.is_like_msvc
}
