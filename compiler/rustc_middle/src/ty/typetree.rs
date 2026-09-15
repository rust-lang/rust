use rustc_ast::expand::typetree::{FncTree, Kind, Type, TypeTree};
use tracing::trace;

use crate::ty::context::TyCtxt;
use crate::ty::{self, Ty};

/// Generate TypeTree information for autodiff.
/// This function creates TypeTree metadata that describes the memory layout
/// of function parameters and return types for Enzyme autodiff.
pub fn fnc_typetrees<'tcx>(tcx: TyCtxt<'tcx>, fn_ty: Ty<'tcx>) -> FncTree {
    // Check if TypeTrees are disabled via NoTT flag
    if tcx.sess.opts.unstable_opts.autodiff.contains(&rustc_session::config::AutoDiff::NoTT) {
        return FncTree { args: vec![], ret: TypeTree::new() };
    }

    // Check if this is actually a function type
    if !fn_ty.is_fn() {
        return FncTree { args: vec![], ret: TypeTree::new() };
    }

    // Get the function signature
    let fn_sig = fn_ty.fn_sig(tcx);
    let sig = tcx.instantiate_bound_regions_with_erased(fn_sig);

    // Create TypeTrees for each input parameter
    let mut args = vec![];
    for ty in sig.inputs().iter() {
        let type_tree = typetree_from_ty(tcx, *ty);
        args.push(type_tree);
    }

    // Create TypeTree for return type
    let ret = typetree_from_ty(tcx, sig.output());

    let f = FncTree { args, ret };
    f
}

/// Generate a TypeTree for a specific type.
/// Mainly a convenience wrapper around the actual implementation.
pub fn typetree_from_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> TypeTree {
    if !tcx.sess.opts.unstable_opts.autodiff.contains(&rustc_session::config::AutoDiff::Enable) {
        return TypeTree::new();
    }
    if tcx.sess.opts.unstable_opts.autodiff.contains(&rustc_session::config::AutoDiff::NoTT) {
        return TypeTree::new();
    }
    let mut visited = Vec::new();
    typetree_from_ty_impl_inner(tcx, ty, 0, &mut visited, false)
}

/// Maximum recursion depth for TypeTree generation to prevent stack overflow
/// from pathological deeply nested types. Combined with cycle detection.
const MAX_TYPETREE_DEPTH: usize = 6;

fn handle_indirection<'a>(
    ty: Ty<'a>,
    tcx: TyCtxt<'a>,
    depth: usize,
    visited: &mut Vec<Ty<'a>>,
) -> TypeTree {
    let Some(inner_ty) = ty.builtin_deref(true) else {
        bug!("incorrect autodiff typetree handling for type: {}", ty);
    };
    // A pointer to a slice-tailed DST is a fat pointer `{data, len}`. `RustSlice` describes both
    // LLVM arguments, while its child describes the memory reached through `data`.
    let typing_env = ty::TypingEnv::fully_monomorphized();
    if let ty::Slice(element_ty) = tcx.struct_tail_for_codegen(inner_ty, typing_env).kind() {
        // `layout.size` here is the sized prefix of `inner_ty`, not the slice element size.
        // Direct slices, transparent wrappers (`OsStr`), and ZST-prefixed DSTs have no byte
        // offset to preserve. Nonzero prefixes (e.g. `Header<[f32]>`) keep field offsets.
        // ZST elements still take this path and yield an empty child TypeTree (size 0).
        let child = if tcx
            .layout_of(typing_env.as_query_input(inner_ty))
            .is_ok_and(|layout| layout.size.bytes() == 0)
        {
            typetree_from_ty_impl_inner(tcx, *element_ty, depth + 1, visited, false)
        } else {
            typetree_from_ty_impl_inner(tcx, inner_ty, depth + 1, visited, true)
        };
        return TypeTree(vec![Type {
            offset: -1,
            size: tcx.data_layout.pointer_size().bytes_usize(),
            kind: Kind::RustSlice,
            child,
        }]);
    }

    let child = typetree_from_ty_impl_inner(tcx, inner_ty, depth + 1, visited, true);
    return TypeTree(vec![Type {
        offset: -1,
        size: tcx.data_layout.pointer_size().bytes_usize(),
        kind: Kind::Pointer,
        child,
    }]);
}

/// Internal implementation with context about whether this is for a reference target.
fn typetree_from_ty_impl_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    depth: usize,
    visited: &mut Vec<Ty<'tcx>>,
    is_reference_target: bool,
) -> TypeTree {
    if depth >= MAX_TYPETREE_DEPTH {
        trace!("typetree depth limit {} reached for type: {}", MAX_TYPETREE_DEPTH, ty);
        return TypeTree::new();
    }

    if visited.contains(&ty) {
        return TypeTree::new();
    }
    visited.push(ty);

    let tree = match ty.kind() {
        // Direct slices are handled by `handle_indirection`. This arm describes a slice tail while
        // recursing through a prefixed DST, so its caller can add the field offset.
        ty::Slice(element_ty) => {
            typetree_from_ty_impl_inner(tcx, *element_ty, depth + 1, visited, false)
        }
        ty::Ref(..) | ty::RawPtr(..) => handle_indirection(ty, tcx, depth, visited),
        ty::Adt(def, _) if def.is_box() => handle_indirection(ty, tcx, depth, visited),
        ty::Array(element_ty, len_const) => {
            // Lengths are normalized by callers (struct fields via
            // `normalize_erasing_regions`); keep a graceful fallback rather
            // than asserting when evaluation still fails.
            let len = len_const.try_to_target_usize(tcx).unwrap_or_else(|| {
                trace!(
                    "autodiff typetree: array length {len_const:?} not evaluable; \
                     emitting empty TypeTree"
                );
                0
            });
            if len == 0 {
                TypeTree::new()
            } else {
                let element_tree =
                    typetree_from_ty_impl_inner(tcx, *element_ty, depth + 1, visited, false);
                let mut types = Vec::new();
                for elem_type in &element_tree.0 {
                    types.push(Type::from_ty(-1, elem_type));
                }

                TypeTree(types)
            }
        }
        ty::Tuple(tuple_types) => {
            if tuple_types.is_empty() {
                TypeTree::new()
            } else {
                let mut types = Vec::new();
                let mut current_offset = 0;

                for tuple_ty in tuple_types.iter() {
                    let element_tree =
                        typetree_from_ty_impl_inner(tcx, tuple_ty, depth + 1, visited, false);

                    let element_layout = tcx
                        .layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(tuple_ty))
                        .ok()
                        .map(|layout| layout.size.bytes_usize())
                        .unwrap_or(0);

                    for elem_type in &element_tree.0 {
                        let offset = if elem_type.offset == -1 {
                            current_offset as isize
                        } else {
                            current_offset as isize + elem_type.offset
                        };
                        types.push(Type::from_ty(offset, elem_type));
                    }

                    current_offset += element_layout;
                }

                TypeTree(types)
            }
        }
        ty::Adt(adt_def, args) if adt_def.is_struct() => {
            let typing_env = ty::TypingEnv::fully_monomorphized();
            let struct_layout = tcx.layout_of(typing_env.as_query_input(ty));
            if let Ok(layout) = struct_layout {
                let mut types = Vec::new();

                for (field_idx, field_def) in adt_def.all_fields().enumerate() {
                    // `FieldDef::ty` returns `Unnormalized`; normalize before recursion so
                    // array lengths / projections are concrete for `struct_tail_for_codegen`
                    // and `try_to_target_usize` (see #160635).
                    let field_ty =
                        tcx.normalize_erasing_regions(typing_env, field_def.ty(tcx, args));
                    let field_tree =
                        typetree_from_ty_impl_inner(tcx, field_ty, depth + 1, visited, false);

                    let field_offset = layout.fields.offset(field_idx).bytes_usize();

                    for elem_type in &field_tree.0 {
                        let offset = if elem_type.offset == -1 {
                            field_offset as isize
                        } else {
                            field_offset as isize + elem_type.offset
                        };
                        types.push(Type::from_ty(offset, elem_type));
                    }
                }

                TypeTree(types)
            } else {
                TypeTree::new()
            }
        }
        ty::Char | ty::Bool | ty::Infer(ty::IntVar(_)) | ty::Int(_) | ty::Uint(_) => {
            let kind = Kind::Integer;
            let size = ty.primitive_size(tcx).bytes_usize();
            let offset = if is_reference_target { 0 } else { -1 };
            TypeTree(vec![Type { offset, size, kind, child: TypeTree::new() }])
        }
        ty::Float(_) | ty::Infer(ty::FloatVar(_)) => {
            let (enzyme_ty, size) = match ty {
                x if x == tcx.types.f16 => (Kind::Half, 2),
                x if x == tcx.types.f32 => (Kind::Float, 4),
                x if x == tcx.types.f64 => (Kind::Double, 8),
                x if x == tcx.types.f128 => (Kind::F128, 16),
                _ => bug!("Unexpected floating point type: {:?}", ty),
            };
            let offset = if is_reference_target { 0 } else { -1 };
            TypeTree(vec![Type { offset, size, kind: enzyme_ty, child: TypeTree::new() }])
        }
        _ => TypeTree::new(),
    };

    let popped = visited.pop();
    debug_assert_eq!(popped, Some(ty));
    tree
}
