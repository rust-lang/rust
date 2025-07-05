use rustc_ast as ast;
use rustc_ast::FnRetTy;
use rustc_ast::expand::typetree::{Type, Kind, TypeTree, FncTree};
use rustc_middle::ty::{Ty, TyCtxt, ParamEnv, ParamEnvAnd, Adt};
use rustc_middle::ty::layout::{FieldsShape, LayoutOf};
use rustc_middle::hir;
use rustc_span::Span;
use rustc_ast::expand::autodiff_attrs::DiffActivity;

#[cfg(llvm_enzyme)]
pub fn typetree_from<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> TypeTree {
    let mut visited = vec![];
    let ty = typetree_from_ty(ty, tcx, 0, false, &mut visited, None);
    let tt = Type { offset: -1, kind: Kind::Pointer, size: 8, child: ty };
    return TypeTree(vec![tt]);
}

// This function combines three tasks. To avoid traversing each type 3x, we combine them.
// 1. Create a TypeTree from a Ty. This is the main task.
// 2. IFF da is not empty, we also want to adjust DiffActivity to account for future MIR->LLVM
//    lowering. E.g. fat ptr are going to introduce an extra int.
// 3. IFF da is not empty, we are creating TT for a function directly differentiated (has an
//    autodiff macro on top). Here we want to make sure that shadows are mutable internally.
//    We know the outermost ref/ptr indirection is mutability - we generate it like that.
//    We now have to make sure that inner ptr/ref are mutable too, or issue a warning.
//    Not an error, becaues it only causes issues if they are actually read, which we don't check
//    yet. We should add such analysis to relibably either issue an error or accept without warning.
//    If there only were some reasearch to do that...
#[cfg(llvm_enzyme)]
pub fn fnc_typetrees<'tcx>(tcx: TyCtxt<'tcx>, fn_ty: Ty<'tcx>, da: &mut Vec<DiffActivity>, span: Option<Span>) -> FncTree {
    if !fn_ty.is_fn() {
        return FncTree { args: vec![], ret: TypeTree::new() };
    }
    let fnc_binder: ty::Binder<'_, ty::FnSig<'_>> = fn_ty.fn_sig(tcx);

    // If rustc compiles the unmodified primal, we know that this copy of the function
    // also has correct lifetimes. We know that Enzyme won't free the shadow too early
    // (or actually at all), so let's strip lifetimes when computing the layout.
    // Recommended by compiler-errors:
    // https://discord.com/channels/273534239310479360/957720175619215380/1223454360676208751
    let x = tcx.instantiate_bound_regions_with_erased(fnc_binder);

    let mut new_activities = vec![];
    let mut new_positions = vec![];
    let mut visited = vec![];
    let mut args = vec![];
    for (i, ty) in x.inputs().iter().enumerate() {
        // We care about safety checks, if an argument get's duplicated and we write into the
        // shadow. That's equivalent to Duplicated or DuplicatedOnly.
        let safety = if !da.is_empty() {
            assert!(da.len() == x.inputs().len(), "{:?} != {:?}", da.len(), x.inputs().len());
            // If we have Activities, we also have spans
            assert!(span.is_some());
            match da[i] {
                DiffActivity::DuplicatedOnly | DiffActivity::Duplicated => true,
                _ => false,
            }
        } else {
            false
        };

        visited.clear();
        if ty.is_unsafe_ptr() || ty.is_ref() || ty.is_box() {
            if ty.is_fn_ptr() {
                unimplemented!("what to do whith fn ptr?");
            }
            let inner_ty = ty.builtin_deref(true).unwrap().ty;
            if inner_ty.is_slice() {
                // We know that the lenght will be passed as extra arg.
                let child = typetree_from_ty(inner_ty, tcx, 1, safety, &mut visited, span);
                let tt = Type { offset: -1, kind: Kind::Pointer, size: 8, child };
                args.push(TypeTree(vec![tt]));
                let i64_tt = Type { offset: -1, kind: Kind::Integer, size: 8, child: TypeTree::new() };
                args.push(TypeTree(vec![i64_tt]));
                if !da.is_empty() {
                    // We are looking at a slice. The length of that slice will become an
                    // extra integer on llvm level. Integers are always const.
                    // However, if the slice get's duplicated, we want to know to later check the
                    // size. So we mark the new size argument as FakeActivitySize.
                    let activity = match da[i] {
                        DiffActivity::DualOnly | DiffActivity::Dual |
                            DiffActivity::DuplicatedOnly | DiffActivity::Duplicated
                            => DiffActivity::FakeActivitySize,
                        DiffActivity::Const => DiffActivity::Const,
                        _ => panic!("unexpected activity for ptr/ref"),
                    };
                    new_activities.push(activity);
                    new_positions.push(i + 1);
                }
                trace!("ABI MATCHING!");
                continue;
            }
        }
        let arg_tt = typetree_from_ty(*ty, tcx, 0, safety, &mut visited, span);
        args.push(arg_tt);
    }

    // now add the extra activities coming from slices
    // Reverse order to not invalidate the indices
    for _ in 0..new_activities.len() {
        let pos = new_positions.pop().unwrap();
        let activity = new_activities.pop().unwrap();
        da.insert(pos, activity);
    }

    visited.clear();
    let ret = typetree_from_ty(x.output(), tcx, 0, false, &mut visited, span);

    FncTree { args, ret }
}


// Error type for warnings
#[derive(Debug)]
pub struct AutodiffUnsafeInnerConstRef {
    pub span: Span,
    pub ty: String,
}

#[cfg(llvm_enzyme)]
fn typetree_from_ty<'a>(ty: Ty<'a>, tcx: TyCtxt<'a>, depth: usize, safety: bool, visited: &mut Vec<Ty<'a>>, span: Option<Span>) -> TypeTree {
    if depth > 20 {
        trace!("depth > 20 for ty: {}", &ty);
    }
    if visited.contains(&ty) {
        // recursive type
        trace!("recursive type: {}", &ty);
        return TypeTree::new();
    }
    visited.push(ty);

    if ty.is_unsafe_ptr() || ty.is_ref() || ty.is_box() {
        if ty.is_fn_ptr() {
            unimplemented!("what to do whith fn ptr?");
        }

        let inner_ty_and_mut = ty.builtin_deref(true).unwrap();
        let is_mut = inner_ty_and_mut.mutbl == hir::Mutability::Mut;
        let inner_ty = inner_ty_and_mut.ty;

        // Now account for inner mutability.
        if !is_mut && depth > 0 && safety {
            let ptr_ty: String = if ty.is_ref() {
                "ref"
            } else if ty.is_unsafe_ptr() {
                "ptr"
            } else {
                assert!(ty.is_box());
                "box"
            }.to_string();

            // If we have mutability, we also have a span
            assert!(span.is_some());
            let span = span.unwrap();

            tcx.sess
            .dcx()
            .emit_warning(AutodiffUnsafeInnerConstRef{span, ty: ptr_ty});
        }

        let child = typetree_from_ty(inner_ty, tcx, depth + 1, safety, visited, span);
        let tt = Type { offset: -1, kind: Kind::Pointer, size: 8, child };
        visited.pop();
        return TypeTree(vec![tt]);
    }

    if ty.is_closure() || ty.is_coroutine() || ty.is_fresh() || ty.is_fn() {
        visited.pop();
        return TypeTree::new();
    }

    if ty.is_scalar() {
        let (kind, size) = if ty.is_integral() || ty.is_char() || ty.is_bool() {
            (Kind::Integer, ty.primitive_size(tcx).bytes_usize())
        } else if ty.is_floating_point() {
            match ty {
                x if x == tcx.types.f32 => (Kind::Float, 4),
                x if x == tcx.types.f64 => (Kind::Double, 8),
                _ => panic!("floatTy scalar that is neither f32 nor f64"),
            }
        } else {
            panic!("scalar that is neither integral nor floating point");
        };
        visited.pop();
        return TypeTree(vec![Type { offset: -1, child: TypeTree::new(), kind, size }]);
    }

    let param_env_and = ParamEnvAnd { param_env: ParamEnv::empty(), value: ty };

    let layout = tcx.layout_of(param_env_and);
    assert!(layout.is_ok());

    let layout = layout.unwrap().layout;
    let fields = layout.fields();
    let max_size = layout.size();

    if ty.is_adt() && !ty.is_simd() {
        let adt_def = ty.ty_adt_def().unwrap();

        if adt_def.is_struct() {
            let (offsets, _memory_index) = match fields {
                // Manuel TODO:
                FieldsShape::Arbitrary { offsets: o, memory_index: m } => (o, m),
                FieldsShape::Array { .. } => {return TypeTree::new();}, //e.g. core::arch::x86_64::__m128i, TODO: later
                FieldsShape::Union(_) => {return TypeTree::new();},
                FieldsShape::Primitive => {return TypeTree::new();},
            };

            let substs = match ty.kind() {
                Adt(_, subst_ref) => subst_ref,
                _ => panic!(""),
            };

            let fields = adt_def.all_fields();
            let fields = fields
                .into_iter()
                .zip(offsets.into_iter())
                .filter_map(|(field, offset)| {
                    let field_ty: Ty<'_> = field.ty(tcx, substs);
                    let field_ty: Ty<'_> =
                        tcx.normalize_erasing_regions(ParamEnv::empty(), field_ty);

                    if field_ty.is_phantom_data() {
                        return None;
                    }

                    let mut child = typetree_from_ty(field_ty, tcx, depth + 1, safety, visited, span).0;

                    for c in &mut child {
                        if c.offset == -1 {
                            c.offset = offset.bytes() as isize
                        } else {
                            c.offset += offset.bytes() as isize;
                        }
                    }

                    Some(child)
                })
                .flatten()
                .collect::<Vec<Type>>();

            visited.pop();
            let ret_tt = TypeTree(fields);
            return ret_tt;
        } else if adt_def.is_enum() {
            // Enzyme can't represent enums, so let it figure it out itself, without seeeding
            // typetree
            //unimplemented!("adt that is an enum");
        } else {
            //let ty_name = tcx.def_path_debug_str(adt_def.did());
            //tcx.sess.emit_fatal(UnsupportedUnion { ty_name });
        }
    }

    if ty.is_simd() {
        trace!("simd");
        let (_size, inner_ty) = ty.simd_size_and_type(tcx);
        let _sub_tt = typetree_from_ty(inner_ty, tcx, depth + 1, safety, visited, span);
        // TODO
        visited.pop();
        return TypeTree::new();
    }

    if ty.is_array() {
        let (stride, count) = match fields {
            FieldsShape::Array { stride: s, count: c } => (s, c),
            _ => panic!(""),
        };
        let byte_stride = stride.bytes_usize();
        let byte_max_size = max_size.bytes_usize();

        assert!(byte_stride * *count as usize == byte_max_size);
        if (*count as usize) == 0 {
            return TypeTree::new();
        }
        let sub_ty = ty.builtin_index().unwrap();
        let subtt = typetree_from_ty(sub_ty, tcx, depth + 1, safety, visited, span);

        // calculate size of subtree
        let param_env_and = ParamEnvAnd { param_env: ParamEnv::empty(), value: sub_ty };
        let size = tcx.layout_of(param_env_and).unwrap().size.bytes() as usize;
        let tt = TypeTree(
            std::iter::repeat(subtt)
                .take(*count as usize)
                .enumerate()
                .map(|(idx, x)| x.0.into_iter().map(move |x| x.add_offset((idx * size) as isize)))
                .flatten()
                .collect(),
        );

        visited.pop();
        return tt;
    }

    if ty.is_slice() {
        let sub_ty = ty.builtin_index().unwrap();
        let subtt = typetree_from_ty(sub_ty, tcx, depth + 1, safety, visited, span);

        visited.pop();
        return subtt;
    }

    visited.pop();
    TypeTree::new()
}

// AST-based type tree construction (simplified fallback)
#[cfg(llvm_enzyme)]
pub fn construct_typetree_from_ty(ty: &ast::Ty) -> TypeTree {
    // For now, return empty type tree to let Enzyme figure out layout
    // In a full implementation, we'd need to convert AST types to Ty<'tcx>
    // and use the layout-based approach from the old code
    TypeTree::new()
}

#[cfg(llvm_enzyme)]
pub fn construct_typetree_from_fnsig(sig: &ast::FnSig) -> (Vec<TypeTree>, TypeTree) {
    // For now, return empty type trees
    // This will be replaced with proper layout-based construction
    let inputs: Vec<TypeTree> = sig.decl.inputs.iter()
        .map(|_| TypeTree::new())
        .collect();
    
    let output = match &sig.decl.output {
        FnRetTy::Default(_) => TypeTree::new(),
        FnRetTy::Ty(_) => TypeTree::new(),
    };
    
    (inputs, output)
}
