use rustc_ast::expand::typetree::{FncTree, Kind, Type, TypeTree};
use crate::ty::{self, Ty, context::TyCtxt};
use tracing::trace;

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

    FncTree { args, ret }
}

/// Generate TypeTree for a specific type.
/// This function analyzes a Rust type and creates appropriate TypeTree metadata.
pub fn typetree_from_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> TypeTree {
    let mut visited = Vec::new();
    typetree_from_ty_inner(tcx, ty, 0, &mut visited)
}

/// Maximum recursion depth for TypeTree generation to prevent stack overflow
/// from pathological deeply nested types. Combined with cycle detection.
const MAX_TYPETREE_DEPTH: usize = 6;

/// Internal recursive function for TypeTree generation with cycle detection and depth limiting.
fn typetree_from_ty_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    depth: usize,
    visited: &mut Vec<Ty<'tcx>>,
) -> TypeTree {
    if depth >= MAX_TYPETREE_DEPTH {
        trace!("typetree depth limit {} reached for type: {}", MAX_TYPETREE_DEPTH, ty);
        return TypeTree::new();
    }

    if visited.contains(&ty) {
        return TypeTree::new();
    }

    visited.push(ty);
    let result = typetree_from_ty_impl(tcx, ty, depth, visited);
    visited.pop();
    result
}

/// Implementation of TypeTree generation logic.
fn typetree_from_ty_impl<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    depth: usize,
    visited: &mut Vec<Ty<'tcx>>,
) -> TypeTree {
    typetree_from_ty_impl_inner(tcx, ty, depth, visited, false)
}

/// Internal implementation with context about whether this is for a reference target.
fn typetree_from_ty_impl_inner<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    depth: usize,
    visited: &mut Vec<Ty<'tcx>>,
    is_reference_target: bool,
) -> TypeTree {
    if ty.is_scalar() {
        let (kind, size) = if ty.is_integral() || ty.is_char() || ty.is_bool() {
            (Kind::Integer, ty.primitive_size(tcx).bytes_usize())
        } else if ty.is_floating_point() {
            match ty {
                x if x == tcx.types.f16 => (Kind::Half, 2),
                x if x == tcx.types.f32 => (Kind::Float, 4),
                x if x == tcx.types.f64 => (Kind::Double, 8),
                x if x == tcx.types.f128 => (Kind::F128, 16),
                _ => (Kind::Integer, 0),
            }
        } else {
            (Kind::Integer, 0)
        };

        // Use offset 0 for scalars that are direct targets of references (like &f64)
        // Use offset -1 for scalars used directly (like function return types)
        let offset = if is_reference_target && !ty.is_array() { 0 } else { -1 };
        return TypeTree(vec![Type { offset, size, kind, child: TypeTree::new() }]);
    }

    if ty.is_ref() || ty.is_raw_ptr() || ty.is_box() {
        let Some(inner_ty) = ty.builtin_deref(true) else {
            return TypeTree::new();
        };

        let child = typetree_from_ty_impl_inner(tcx, inner_ty, depth + 1, visited, true);
        return TypeTree(vec![Type {
            offset: -1,
            size: tcx.data_layout.pointer_size().bytes_usize(),
            kind: Kind::Pointer,
            child,
        }]);
    }

    if ty.is_array() {
        if let ty::Array(element_ty, len_const) = ty.kind() {
            let len = len_const.try_to_target_usize(tcx).unwrap_or(0);
            if len == 0 {
                return TypeTree::new();
            }
            let element_tree =
                typetree_from_ty_impl_inner(tcx, *element_ty, depth + 1, visited, false);
            let mut types = Vec::new();
            for elem_type in &element_tree.0 {
                types.push(Type {
                    offset: -1,
                    size: elem_type.size,
                    kind: elem_type.kind,
                    child: elem_type.child.clone(),
                });
            }

            return TypeTree(types);
        }
    }

    if ty.is_slice() {
        if let ty::Slice(element_ty) = ty.kind() {
            let element_tree =
                typetree_from_ty_impl_inner(tcx, *element_ty, depth + 1, visited, false);
            return element_tree;
        }
    }

    if let ty::Tuple(tuple_types) = ty.kind() {
        if tuple_types.is_empty() {
            return TypeTree::new();
        }

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
                types.push(Type {
                    offset: if elem_type.offset == -1 {
                        current_offset as isize
                    } else {
                        current_offset as isize + elem_type.offset
                    },
                    size: elem_type.size,
                    kind: elem_type.kind,
                    child: elem_type.child.clone(),
                });
            }

            current_offset += element_layout;
        }

        return TypeTree(types);
    }

    if let ty::Adt(adt_def, args) = ty.kind() {
        if adt_def.is_struct() {
            let struct_layout =
                tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty));
            if let Ok(layout) = struct_layout {
                let mut types = Vec::new();

                for (field_idx, field_def) in adt_def.all_fields().enumerate() {
                    let field_ty = field_def.ty(tcx, args);
                    let field_tree = typetree_from_ty_impl_inner(
                        tcx,
                        field_ty.skip_norm_wip(),
                        depth + 1,
                        visited,
                        false,
                    );

                    let field_offset = layout.fields.offset(field_idx).bytes_usize();

                    for elem_type in &field_tree.0 {
                        types.push(Type {
                            offset: if elem_type.offset == -1 {
                                field_offset as isize
                            } else {
                                field_offset as isize + elem_type.offset
                            },
                            size: elem_type.size,
                            kind: elem_type.kind,
                            child: elem_type.child.clone(),
                        });
                    }
                }

                return TypeTree(types);
            }
        }
    }

    TypeTree::new()
}
