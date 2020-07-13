//! Type-checking for the rust-intrinsic and platform-intrinsic
//! intrinsics that the compiler exposes.

use crate::require_same_types;

use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_target::spec::abi::Abi;

use std::iter;

fn equate_intrinsic_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    it: &hir::ForeignItem<'_>,
    def_id: DefId,
    n_tps: usize,
    abi: Abi,
    safety: hir::Unsafety,
    inputs: Vec<Ty<'tcx>>,
    output: Ty<'tcx>,
) {
    match it.kind {
        hir::ForeignItemKind::Fn(..) => {}
        _ => {
            struct_span_err!(tcx.sess, it.span, E0622, "intrinsic must be a function")
                .span_label(it.span, "expected a function")
                .emit();
            return;
        }
    }

    let i_n_tps = tcx.generics_of(def_id).own_counts().types;
    if i_n_tps != n_tps {
        let span = match it.kind {
            hir::ForeignItemKind::Fn(_, _, ref generics) => generics.span,
            _ => bug!(),
        };

        struct_span_err!(
            tcx.sess,
            span,
            E0094,
            "intrinsic has wrong number of type \
                         parameters: found {}, expected {}",
            i_n_tps,
            n_tps
        )
        .span_label(span, format!("expected {} type parameter", n_tps))
        .emit();
        return;
    }

    let fty = tcx.mk_fn_ptr(ty::Binder::bind(tcx.mk_fn_sig(
        inputs.into_iter(),
        output,
        false,
        safety,
        abi,
    )));
    let cause = ObligationCause::new(it.span, it.hir_id, ObligationCauseCode::IntrinsicType);
    require_same_types(tcx, &cause, tcx.mk_fn_ptr(tcx.fn_sig(def_id)), fty);
}

/// Returns `true` if the given intrinsic is unsafe to call or not.
pub fn intrinsic_operation_unsafety(intrinsic: Symbol) -> hir::Unsafety {
    match intrinsic {
        sym::abort
        | sym::size_of
        | sym::min_align_of
        | sym::needs_drop
        | sym::caller_location
        | sym::size_of_val
        | sym::min_align_of_val
        | sym::add_with_overflow
        | sym::sub_with_overflow
        | sym::mul_with_overflow
        | sym::wrapping_add
        | sym::wrapping_sub
        | sym::wrapping_mul
        | sym::saturating_add
        | sym::saturating_sub
        | sym::rotate_left
        | sym::rotate_right
        | sym::ctpop
        | sym::ctlz
        | sym::cttz
        | sym::bswap
        | sym::bitreverse
        | sym::discriminant_value
        | sym::type_id
        | sym::likely
        | sym::unlikely
        | sym::ptr_guaranteed_eq
        | sym::ptr_guaranteed_ne
        | sym::minnumf32
        | sym::minnumf64
        | sym::maxnumf32
        | sym::rustc_peek
        | sym::maxnumf64
        | sym::type_name
        | sym::variant_count => hir::Unsafety::Normal,
        _ => hir::Unsafety::Unsafe,
    }
}

/// Remember to add all intrinsics here, in librustc_codegen_llvm/intrinsic.rs,
/// and in libcore/intrinsics.rs
pub fn check_intrinsic_type(tcx: TyCtxt<'_>, it: &hir::ForeignItem<'_>) {
    let param = |n| tcx.mk_ty_param(n, Symbol::intern(&format!("P{}", n)));
    let def_id = tcx.hir().local_def_id(it.hir_id).to_def_id();
    let intrinsic_name = tcx.item_name(def_id);
    let name_str = intrinsic_name.as_str();

    let mk_va_list_ty = |mutbl| {
        tcx.lang_items().va_list().map(|did| {
            let region = tcx.mk_region(ty::ReLateBound(ty::INNERMOST, ty::BrAnon(0)));
            let env_region = ty::ReLateBound(ty::INNERMOST, ty::BrEnv);
            let va_list_ty = tcx.type_of(did).subst(tcx, &[region.into()]);
            (
                tcx.mk_ref(tcx.mk_region(env_region), ty::TypeAndMut { ty: va_list_ty, mutbl }),
                va_list_ty,
            )
        })
    };

    let (n_tps, inputs, output, unsafety) = if name_str.starts_with("atomic_") {
        let split: Vec<&str> = name_str.split('_').collect();
        assert!(split.len() >= 2, "Atomic intrinsic in an incorrect format");

        //We only care about the operation here
        let (n_tps, inputs, output) = match split[1] {
            "cxchg" | "cxchgweak" => (
                1,
                vec![tcx.mk_mut_ptr(param(0)), param(0), param(0)],
                tcx.intern_tup(&[param(0), tcx.types.bool]),
            ),
            "load" => (1, vec![tcx.mk_imm_ptr(param(0))], param(0)),
            "store" => (1, vec![tcx.mk_mut_ptr(param(0)), param(0)], tcx.mk_unit()),

            "xchg" | "xadd" | "xsub" | "and" | "nand" | "or" | "xor" | "max" | "min" | "umax"
            | "umin" => (1, vec![tcx.mk_mut_ptr(param(0)), param(0)], param(0)),
            "fence" | "singlethreadfence" => (0, Vec::new(), tcx.mk_unit()),
            op => {
                struct_span_err!(
                    tcx.sess,
                    it.span,
                    E0092,
                    "unrecognized atomic operation function: `{}`",
                    op
                )
                .span_label(it.span, "unrecognized atomic operation")
                .emit();
                return;
            }
        };
        (n_tps, inputs, output, hir::Unsafety::Unsafe)
    } else {
        let unsafety = intrinsic_operation_unsafety(intrinsic_name);
        let (n_tps, inputs, output) = match intrinsic_name {
            sym::abort => (0, Vec::new(), tcx.types.never),
            sym::unreachable => (0, Vec::new(), tcx.types.never),
            sym::breakpoint => (0, Vec::new(), tcx.mk_unit()),
            sym::size_of | sym::pref_align_of | sym::min_align_of | sym::variant_count => {
                (1, Vec::new(), tcx.types.usize)
            }
            sym::size_of_val | sym::min_align_of_val => {
                (1, vec![tcx.mk_imm_ptr(param(0))], tcx.types.usize)
            }
            sym::rustc_peek => (1, vec![param(0)], param(0)),
            sym::caller_location => (0, vec![], tcx.caller_location_ty()),
            sym::assert_inhabited | sym::assert_zero_valid | sym::assert_uninit_valid => {
                (1, Vec::new(), tcx.mk_unit())
            }
            sym::forget => (1, vec![param(0)], tcx.mk_unit()),
            sym::transmute => (2, vec![param(0)], param(1)),
            sym::move_val_init => (1, vec![tcx.mk_mut_ptr(param(0)), param(0)], tcx.mk_unit()),
            sym::prefetch_read_data
            | sym::prefetch_write_data
            | sym::prefetch_read_instruction
            | sym::prefetch_write_instruction => (
                1,
                vec![
                    tcx.mk_ptr(ty::TypeAndMut { ty: param(0), mutbl: hir::Mutability::Not }),
                    tcx.types.i32,
                ],
                tcx.mk_unit(),
            ),
            sym::drop_in_place => (1, vec![tcx.mk_mut_ptr(param(0))], tcx.mk_unit()),
            sym::needs_drop => (1, Vec::new(), tcx.types.bool),

            sym::type_name => (1, Vec::new(), tcx.mk_static_str()),
            sym::type_id => (1, Vec::new(), tcx.types.u64),
            sym::offset | sym::arith_offset => (
                1,
                vec![
                    tcx.mk_ptr(ty::TypeAndMut { ty: param(0), mutbl: hir::Mutability::Not }),
                    tcx.types.isize,
                ],
                tcx.mk_ptr(ty::TypeAndMut { ty: param(0), mutbl: hir::Mutability::Not }),
            ),
            sym::copy | sym::copy_nonoverlapping => (
                1,
                vec![
                    tcx.mk_ptr(ty::TypeAndMut { ty: param(0), mutbl: hir::Mutability::Not }),
                    tcx.mk_ptr(ty::TypeAndMut { ty: param(0), mutbl: hir::Mutability::Mut }),
                    tcx.types.usize,
                ],
                tcx.mk_unit(),
            ),
            sym::volatile_copy_memory | sym::volatile_copy_nonoverlapping_memory => (
                1,
                vec![
                    tcx.mk_ptr(ty::TypeAndMut { ty: param(0), mutbl: hir::Mutability::Mut }),
                    tcx.mk_ptr(ty::TypeAndMut { ty: param(0), mutbl: hir::Mutability::Not }),
                    tcx.types.usize,
                ],
                tcx.mk_unit(),
            ),
            sym::write_bytes | sym::volatile_set_memory => (
                1,
                vec![
                    tcx.mk_ptr(ty::TypeAndMut { ty: param(0), mutbl: hir::Mutability::Mut }),
                    tcx.types.u8,
                    tcx.types.usize,
                ],
                tcx.mk_unit(),
            ),
            sym::sqrtf32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::sqrtf64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::powif32 => (0, vec![tcx.types.f32, tcx.types.i32], tcx.types.f32),
            sym::powif64 => (0, vec![tcx.types.f64, tcx.types.i32], tcx.types.f64),
            sym::sinf32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::sinf64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::cosf32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::cosf64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::powf32 => (0, vec![tcx.types.f32, tcx.types.f32], tcx.types.f32),
            sym::powf64 => (0, vec![tcx.types.f64, tcx.types.f64], tcx.types.f64),
            sym::expf32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::expf64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::exp2f32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::exp2f64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::logf32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::logf64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::log10f32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::log10f64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::log2f32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::log2f64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::fmaf32 => (0, vec![tcx.types.f32, tcx.types.f32, tcx.types.f32], tcx.types.f32),
            sym::fmaf64 => (0, vec![tcx.types.f64, tcx.types.f64, tcx.types.f64], tcx.types.f64),
            sym::fabsf32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::fabsf64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::minnumf32 => (0, vec![tcx.types.f32, tcx.types.f32], tcx.types.f32),
            sym::minnumf64 => (0, vec![tcx.types.f64, tcx.types.f64], tcx.types.f64),
            sym::maxnumf32 => (0, vec![tcx.types.f32, tcx.types.f32], tcx.types.f32),
            sym::maxnumf64 => (0, vec![tcx.types.f64, tcx.types.f64], tcx.types.f64),
            sym::copysignf32 => (0, vec![tcx.types.f32, tcx.types.f32], tcx.types.f32),
            sym::copysignf64 => (0, vec![tcx.types.f64, tcx.types.f64], tcx.types.f64),
            sym::floorf32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::floorf64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::ceilf32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::ceilf64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::truncf32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::truncf64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::rintf32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::rintf64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::nearbyintf32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::nearbyintf64 => (0, vec![tcx.types.f64], tcx.types.f64),
            sym::roundf32 => (0, vec![tcx.types.f32], tcx.types.f32),
            sym::roundf64 => (0, vec![tcx.types.f64], tcx.types.f64),

            sym::volatile_load | sym::unaligned_volatile_load => {
                (1, vec![tcx.mk_imm_ptr(param(0))], param(0))
            }
            sym::volatile_store | sym::unaligned_volatile_store => {
                (1, vec![tcx.mk_mut_ptr(param(0)), param(0)], tcx.mk_unit())
            }

            sym::ctpop
            | sym::ctlz
            | sym::ctlz_nonzero
            | sym::cttz
            | sym::cttz_nonzero
            | sym::bswap
            | sym::bitreverse => (1, vec![param(0)], param(0)),

            sym::add_with_overflow | sym::sub_with_overflow | sym::mul_with_overflow => {
                (1, vec![param(0), param(0)], tcx.intern_tup(&[param(0), tcx.types.bool]))
            }

            sym::ptr_guaranteed_eq | sym::ptr_guaranteed_ne => {
                (1, vec![tcx.mk_imm_ptr(param(0)), tcx.mk_imm_ptr(param(0))], tcx.types.bool)
            }

            sym::ptr_offset_from => {
                (1, vec![tcx.mk_imm_ptr(param(0)), tcx.mk_imm_ptr(param(0))], tcx.types.isize)
            }
            sym::unchecked_div | sym::unchecked_rem | sym::exact_div => {
                (1, vec![param(0), param(0)], param(0))
            }
            sym::unchecked_shl | sym::unchecked_shr | sym::rotate_left | sym::rotate_right => {
                (1, vec![param(0), param(0)], param(0))
            }
            sym::unchecked_add | sym::unchecked_sub | sym::unchecked_mul => {
                (1, vec![param(0), param(0)], param(0))
            }
            sym::wrapping_add | sym::wrapping_sub | sym::wrapping_mul => {
                (1, vec![param(0), param(0)], param(0))
            }
            sym::saturating_add | sym::saturating_sub => (1, vec![param(0), param(0)], param(0)),
            sym::fadd_fast | sym::fsub_fast | sym::fmul_fast | sym::fdiv_fast | sym::frem_fast => {
                (1, vec![param(0), param(0)], param(0))
            }
            sym::float_to_int_unchecked => (2, vec![param(0)], param(1)),

            sym::assume => (0, vec![tcx.types.bool], tcx.mk_unit()),
            sym::likely => (0, vec![tcx.types.bool], tcx.types.bool),
            sym::unlikely => (0, vec![tcx.types.bool], tcx.types.bool),

            sym::discriminant_value => {
                let assoc_items =
                    tcx.associated_items(tcx.lang_items().discriminant_kind_trait().unwrap());
                let discriminant_def_id = assoc_items.in_definition_order().next().unwrap().def_id;

                (
                    1,
                    vec![tcx.mk_imm_ref(
                        tcx.mk_region(ty::ReLateBound(ty::INNERMOST, ty::BrAnon(0))),
                        param(0),
                    )],
                    tcx.mk_projection(discriminant_def_id, tcx.mk_substs([param(0).into()].iter())),
                )
            }

            kw::Try => {
                let mut_u8 = tcx.mk_mut_ptr(tcx.types.u8);
                let try_fn_ty = ty::Binder::bind(tcx.mk_fn_sig(
                    iter::once(mut_u8),
                    tcx.mk_unit(),
                    false,
                    hir::Unsafety::Normal,
                    Abi::Rust,
                ));
                let catch_fn_ty = ty::Binder::bind(tcx.mk_fn_sig(
                    [mut_u8, mut_u8].iter().cloned(),
                    tcx.mk_unit(),
                    false,
                    hir::Unsafety::Normal,
                    Abi::Rust,
                ));
                (
                    0,
                    vec![tcx.mk_fn_ptr(try_fn_ty), mut_u8, tcx.mk_fn_ptr(catch_fn_ty)],
                    tcx.types.i32,
                )
            }

            sym::va_start | sym::va_end => match mk_va_list_ty(hir::Mutability::Mut) {
                Some((va_list_ref_ty, _)) => (0, vec![va_list_ref_ty], tcx.mk_unit()),
                None => bug!("`va_list` language item needed for C-variadic intrinsics"),
            },

            sym::va_copy => match mk_va_list_ty(hir::Mutability::Not) {
                Some((va_list_ref_ty, va_list_ty)) => {
                    let va_list_ptr_ty = tcx.mk_mut_ptr(va_list_ty);
                    (0, vec![va_list_ptr_ty, va_list_ref_ty], tcx.mk_unit())
                }
                None => bug!("`va_list` language item needed for C-variadic intrinsics"),
            },

            sym::va_arg => match mk_va_list_ty(hir::Mutability::Mut) {
                Some((va_list_ref_ty, _)) => (1, vec![va_list_ref_ty], param(0)),
                None => bug!("`va_list` language item needed for C-variadic intrinsics"),
            },

            sym::nontemporal_store => (1, vec![tcx.mk_mut_ptr(param(0)), param(0)], tcx.mk_unit()),

            sym::miri_start_panic => {
                // FIXME - the relevant types aren't lang items,
                // so it's not trivial to check this
                return;
            }

            sym::count_code_region => {
                (0, vec![tcx.types.u32, tcx.types.u32, tcx.types.u32], tcx.mk_unit())
            }

            sym::coverage_counter_add | sym::coverage_counter_subtract => (
                0,
                vec![tcx.types.u32, tcx.types.u32, tcx.types.u32, tcx.types.u32, tcx.types.u32],
                tcx.mk_unit(),
            ),

            sym::coverage_unreachable => (0, vec![tcx.types.u32, tcx.types.u32], tcx.mk_unit()),

            other => {
                struct_span_err!(
                    tcx.sess,
                    it.span,
                    E0093,
                    "unrecognized intrinsic function: `{}`",
                    other,
                )
                .span_label(it.span, "unrecognized intrinsic")
                .emit();
                return;
            }
        };
        (n_tps, inputs, output, unsafety)
    };
    equate_intrinsic_type(tcx, it, def_id, n_tps, Abi::RustIntrinsic, unsafety, inputs, output)
}

/// Type-check `extern "platform-intrinsic" { ... }` functions.
pub fn check_platform_intrinsic_type(tcx: TyCtxt<'_>, it: &hir::ForeignItem<'_>) {
    let param = |n| {
        let name = Symbol::intern(&format!("P{}", n));
        tcx.mk_ty_param(n, name)
    };

    let def_id = tcx.hir().local_def_id(it.hir_id).to_def_id();
    let name = it.ident.as_str();

    let (n_tps, inputs, output) = match &*name {
        "simd_eq" | "simd_ne" | "simd_lt" | "simd_le" | "simd_gt" | "simd_ge" => {
            (2, vec![param(0), param(0)], param(1))
        }
        "simd_add"
        | "simd_sub"
        | "simd_mul"
        | "simd_rem"
        | "simd_div"
        | "simd_shl"
        | "simd_shr"
        | "simd_and"
        | "simd_or"
        | "simd_xor"
        | "simd_fmin"
        | "simd_fmax"
        | "simd_fpow"
        | "simd_saturating_add"
        | "simd_saturating_sub" => (1, vec![param(0), param(0)], param(0)),
        "simd_fsqrt" | "simd_fsin" | "simd_fcos" | "simd_fexp" | "simd_fexp2" | "simd_flog2"
        | "simd_flog10" | "simd_flog" | "simd_fabs" | "simd_floor" | "simd_ceil" => {
            (1, vec![param(0)], param(0))
        }
        "simd_fpowi" => (1, vec![param(0), tcx.types.i32], param(0)),
        "simd_fma" => (1, vec![param(0), param(0), param(0)], param(0)),
        "simd_gather" => (3, vec![param(0), param(1), param(2)], param(0)),
        "simd_scatter" => (3, vec![param(0), param(1), param(2)], tcx.mk_unit()),
        "simd_insert" => (2, vec![param(0), tcx.types.u32, param(1)], param(0)),
        "simd_extract" => (2, vec![param(0), tcx.types.u32], param(1)),
        "simd_cast" => (2, vec![param(0)], param(1)),
        "simd_bitmask" => (2, vec![param(0)], param(1)),
        "simd_select" | "simd_select_bitmask" => (2, vec![param(0), param(1), param(1)], param(1)),
        "simd_reduce_all" | "simd_reduce_any" => (1, vec![param(0)], tcx.types.bool),
        "simd_reduce_add_ordered" | "simd_reduce_mul_ordered" => {
            (2, vec![param(0), param(1)], param(1))
        }
        "simd_reduce_add_unordered"
        | "simd_reduce_mul_unordered"
        | "simd_reduce_and"
        | "simd_reduce_or"
        | "simd_reduce_xor"
        | "simd_reduce_min"
        | "simd_reduce_max"
        | "simd_reduce_min_nanless"
        | "simd_reduce_max_nanless" => (2, vec![param(0)], param(1)),
        name if name.starts_with("simd_shuffle") => match name["simd_shuffle".len()..].parse() {
            Ok(n) => {
                let params = vec![param(0), param(0), tcx.mk_array(tcx.types.u32, n)];
                (2, params, param(1))
            }
            Err(_) => {
                struct_span_err!(
                    tcx.sess,
                    it.span,
                    E0439,
                    "invalid `simd_shuffle`, needs length: `{}`",
                    name
                )
                .emit();
                return;
            }
        },
        _ => {
            let msg = format!("unrecognized platform-specific intrinsic function: `{}`", name);
            tcx.sess.struct_span_err(it.span, &msg).emit();
            return;
        }
    };

    equate_intrinsic_type(
        tcx,
        it,
        def_id,
        n_tps,
        Abi::PlatformIntrinsic,
        hir::Unsafety::Unsafe,
        inputs,
        output,
    )
}
