//! Type-checking for the rust-intrinsic and platform-intrinsic
//! intrinsics that the compiler exposes.

use rustc::traits::{ObligationCause, ObligationCauseCode};
use rustc::ty::{self, TyCtxt, Ty};
use rustc::ty::subst::Subst;
use crate::require_same_types;

use rustc_target::spec::abi::Abi;
use syntax::symbol::InternedString;

use rustc::hir;

use std::iter;

fn equate_intrinsic_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    it: &hir::ForeignItem,
    n_tps: usize,
    abi: Abi,
    safety: hir::Unsafety,
    inputs: Vec<Ty<'tcx>>,
    output: Ty<'tcx>,
) {
    let def_id = tcx.hir().local_def_id_from_hir_id(it.hir_id);

    match it.node {
        hir::ForeignItemKind::Fn(..) => {}
        _ => {
            struct_span_err!(tcx.sess, it.span, E0622,
                             "intrinsic must be a function")
                .span_label(it.span, "expected a function")
                .emit();
            return;
        }
    }

    let i_n_tps = tcx.generics_of(def_id).own_counts().types;
    if i_n_tps != n_tps {
        let span = match it.node {
            hir::ForeignItemKind::Fn(_, _, ref generics) => generics.span,
            _ => bug!()
        };

        struct_span_err!(tcx.sess, span, E0094,
                        "intrinsic has wrong number of type \
                         parameters: found {}, expected {}",
                        i_n_tps, n_tps)
            .span_label(span, format!("expected {} type parameter", n_tps))
            .emit();
        return;
    }

    let fty = tcx.mk_fn_ptr(ty::Binder::bind(tcx.mk_fn_sig(
        inputs.into_iter(),
        output,
        false,
        safety,
        abi
    )));
    let cause = ObligationCause::new(it.span, it.hir_id, ObligationCauseCode::IntrinsicType);
    require_same_types(tcx, &cause, tcx.mk_fn_ptr(tcx.fn_sig(def_id)), fty);
}

/// Returns `true` if the given intrinsic is unsafe to call or not.
pub fn intrisic_operation_unsafety(intrinsic: &str) -> hir::Unsafety {
    match intrinsic {
        "size_of" | "min_align_of" | "needs_drop" |
        "add_with_overflow" | "sub_with_overflow" | "mul_with_overflow" |
        "overflowing_add" | "overflowing_sub" | "overflowing_mul" |
        "saturating_add" | "saturating_sub" |
        "rotate_left" | "rotate_right" |
        "ctpop" | "ctlz" | "cttz" | "bswap" | "bitreverse" |
        "minnumf32" | "minnumf64" | "maxnumf32" | "maxnumf64"
        => hir::Unsafety::Normal,
        _ => hir::Unsafety::Unsafe,
    }
}

/// Remember to add all intrinsics here, in librustc_codegen_llvm/intrinsic.rs,
/// and in libcore/intrinsics.rs
pub fn check_intrinsic_type(tcx: TyCtxt<'_>, it: &hir::ForeignItem) {
    let param = |n| tcx.mk_ty_param(n, InternedString::intern(&format!("P{}", n)));
    let name = it.ident.as_str();

    let mk_va_list_ty = |mutbl| {
        tcx.lang_items().va_list().map(|did| {
            let region = tcx.mk_region(ty::ReLateBound(ty::INNERMOST, ty::BrAnon(0)));
            let env_region = ty::ReLateBound(ty::INNERMOST, ty::BrEnv);
            let va_list_ty = tcx.type_of(did).subst(tcx, &[region.into()]);
            (tcx.mk_ref(tcx.mk_region(env_region), ty::TypeAndMut {
                ty: va_list_ty,
                mutbl
            }), va_list_ty)
        })
    };

    let (n_tps, inputs, output, unsafety) = if name.starts_with("atomic_") {
        let split : Vec<&str> = name.split('_').collect();
        assert!(split.len() >= 2, "Atomic intrinsic in an incorrect format");

        //We only care about the operation here
        let (n_tps, inputs, output) = match split[1] {
            "cxchg" | "cxchgweak" => (1, vec![tcx.mk_mut_ptr(param(0)),
                                              param(0),
                                              param(0)],
                                      tcx.intern_tup(&[param(0), tcx.types.bool])),
            "load" => (1, vec![tcx.mk_imm_ptr(param(0))],
                       param(0)),
            "store" => (1, vec![tcx.mk_mut_ptr(param(0)), param(0)],
                        tcx.mk_unit()),

            "xchg" | "xadd" | "xsub" | "and"  | "nand" | "or" | "xor" | "max" |
            "min"  | "umax" | "umin" => {
                (1, vec![tcx.mk_mut_ptr(param(0)), param(0)],
                 param(0))
            }
            "fence" | "singlethreadfence" => {
                (0, Vec::new(), tcx.mk_unit())
            }
            op => {
                struct_span_err!(tcx.sess, it.span, E0092,
                      "unrecognized atomic operation function: `{}`", op)
                  .span_label(it.span, "unrecognized atomic operation")
                  .emit();
                return;
            }
        };
        (n_tps, inputs, output, hir::Unsafety::Unsafe)
    } else if &name[..] == "abort" || &name[..] == "unreachable" {
        (0, Vec::new(), tcx.types.never, hir::Unsafety::Unsafe)
    } else {
        let unsafety = intrisic_operation_unsafety(&name[..]);
        let (n_tps, inputs, output) = match &name[..] {
            "breakpoint" => (0, Vec::new(), tcx.mk_unit()),
            "size_of" |
            "pref_align_of" | "min_align_of" => (1, Vec::new(), tcx.types.usize),
            "size_of_val" |  "min_align_of_val" => {
                (1, vec![
                    tcx.mk_imm_ref(tcx.mk_region(ty::ReLateBound(ty::INNERMOST,
                                                                 ty::BrAnon(0))),
                                   param(0))
                 ], tcx.types.usize)
            }
            "rustc_peek" => (1, vec![param(0)], param(0)),
            "panic_if_uninhabited" => (1, Vec::new(), tcx.mk_unit()),
            "init" => (1, Vec::new(), param(0)),
            "forget" => (1, vec![param(0)], tcx.mk_unit()),
            "transmute" => (2, vec![ param(0) ], param(1)),
            "move_val_init" => {
                (1,
                 vec![
                    tcx.mk_mut_ptr(param(0)),
                    param(0)
                  ],
               tcx.mk_unit())
            }
            "prefetch_read_data" | "prefetch_write_data" |
            "prefetch_read_instruction" | "prefetch_write_instruction" => {
                (1, vec![tcx.mk_ptr(ty::TypeAndMut {
                          ty: param(0),
                          mutbl: hir::MutImmutable
                         }), tcx.types.i32],
                    tcx.mk_unit())
            }
            "drop_in_place" => {
                (1, vec![tcx.mk_mut_ptr(param(0))], tcx.mk_unit())
            }
            "needs_drop" => (1, Vec::new(), tcx.types.bool),

            "type_name" => (1, Vec::new(), tcx.mk_static_str()),
            "type_id" => (1, Vec::new(), tcx.types.u64),
            "offset" | "arith_offset" => {
              (1,
               vec![
                  tcx.mk_ptr(ty::TypeAndMut {
                      ty: param(0),
                      mutbl: hir::MutImmutable
                  }),
                  tcx.types.isize
               ],
               tcx.mk_ptr(ty::TypeAndMut {
                   ty: param(0),
                   mutbl: hir::MutImmutable
               }))
            }
            "copy" | "copy_nonoverlapping" => {
              (1,
               vec![
                  tcx.mk_ptr(ty::TypeAndMut {
                      ty: param(0),
                      mutbl: hir::MutImmutable
                  }),
                  tcx.mk_ptr(ty::TypeAndMut {
                      ty: param(0),
                      mutbl: hir::MutMutable
                  }),
                  tcx.types.usize,
               ],
               tcx.mk_unit())
            }
            "volatile_copy_memory" | "volatile_copy_nonoverlapping_memory" => {
              (1,
               vec![
                  tcx.mk_ptr(ty::TypeAndMut {
                      ty: param(0),
                      mutbl: hir::MutMutable
                  }),
                  tcx.mk_ptr(ty::TypeAndMut {
                      ty: param(0),
                      mutbl: hir::MutImmutable
                  }),
                  tcx.types.usize,
               ],
               tcx.mk_unit())
            }
            "write_bytes" | "volatile_set_memory" => {
              (1,
               vec![
                  tcx.mk_ptr(ty::TypeAndMut {
                      ty: param(0),
                      mutbl: hir::MutMutable
                  }),
                  tcx.types.u8,
                  tcx.types.usize,
               ],
               tcx.mk_unit())
            }
            "sqrtf32" => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "sqrtf64" => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "powif32" => {
               (0,
                vec![ tcx.types.f32, tcx.types.i32 ],
                tcx.types.f32)
            }
            "powif64" => {
               (0,
                vec![ tcx.types.f64, tcx.types.i32 ],
                tcx.types.f64)
            }
            "sinf32" => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "sinf64" => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "cosf32" => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "cosf64" => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "powf32" => {
               (0,
                vec![ tcx.types.f32, tcx.types.f32 ],
                tcx.types.f32)
            }
            "powf64" => {
               (0,
                vec![ tcx.types.f64, tcx.types.f64 ],
                tcx.types.f64)
            }
            "expf32"   => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "expf64"   => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "exp2f32"  => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "exp2f64"  => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "logf32"   => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "logf64"   => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "log10f32" => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "log10f64" => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "log2f32"  => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "log2f64"  => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "fmaf32" => {
                (0,
                 vec![ tcx.types.f32, tcx.types.f32, tcx.types.f32 ],
                 tcx.types.f32)
            }
            "fmaf64" => {
                (0,
                 vec![ tcx.types.f64, tcx.types.f64, tcx.types.f64 ],
                 tcx.types.f64)
            }
            "fabsf32"      => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "fabsf64"      => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "minnumf32"    => (0, vec![ tcx.types.f32, tcx.types.f32 ], tcx.types.f32),
            "minnumf64"    => (0, vec![ tcx.types.f64, tcx.types.f64 ], tcx.types.f64),
            "maxnumf32"    => (0, vec![ tcx.types.f32, tcx.types.f32 ], tcx.types.f32),
            "maxnumf64"    => (0, vec![ tcx.types.f64, tcx.types.f64 ], tcx.types.f64),
            "copysignf32"  => (0, vec![ tcx.types.f32, tcx.types.f32 ], tcx.types.f32),
            "copysignf64"  => (0, vec![ tcx.types.f64, tcx.types.f64 ], tcx.types.f64),
            "floorf32"     => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "floorf64"     => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "ceilf32"      => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "ceilf64"      => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "truncf32"     => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "truncf64"     => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "rintf32"      => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "rintf64"      => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "nearbyintf32" => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "nearbyintf64" => (0, vec![ tcx.types.f64 ], tcx.types.f64),
            "roundf32"     => (0, vec![ tcx.types.f32 ], tcx.types.f32),
            "roundf64"     => (0, vec![ tcx.types.f64 ], tcx.types.f64),

            "volatile_load" | "unaligned_volatile_load" =>
                (1, vec![ tcx.mk_imm_ptr(param(0)) ], param(0)),
            "volatile_store" | "unaligned_volatile_store" =>
                (1, vec![ tcx.mk_mut_ptr(param(0)), param(0) ], tcx.mk_unit()),

            "ctpop" | "ctlz" | "ctlz_nonzero" | "cttz" | "cttz_nonzero" |
            "bswap" | "bitreverse" =>
                (1, vec![param(0)], param(0)),

            "add_with_overflow" | "sub_with_overflow"  | "mul_with_overflow" =>
                (1, vec![param(0), param(0)],
                tcx.intern_tup(&[param(0), tcx.types.bool])),

            "unchecked_div" | "unchecked_rem" | "exact_div" =>
                (1, vec![param(0), param(0)], param(0)),
            "unchecked_shl" | "unchecked_shr" |
            "rotate_left" | "rotate_right" =>
                (1, vec![param(0), param(0)], param(0)),
            "unchecked_add" | "unchecked_sub" | "unchecked_mul" =>
                (1, vec![param(0), param(0)], param(0)),
            "overflowing_add" | "overflowing_sub" | "overflowing_mul" =>
                (1, vec![param(0), param(0)], param(0)),
            "saturating_add" | "saturating_sub" =>
                (1, vec![param(0), param(0)], param(0)),
            "fadd_fast" | "fsub_fast" | "fmul_fast" | "fdiv_fast" | "frem_fast" =>
                (1, vec![param(0), param(0)], param(0)),

            "assume" => (0, vec![tcx.types.bool], tcx.mk_unit()),
            "likely" => (0, vec![tcx.types.bool], tcx.types.bool),
            "unlikely" => (0, vec![tcx.types.bool], tcx.types.bool),

            "discriminant_value" => (1, vec![
                    tcx.mk_imm_ref(tcx.mk_region(ty::ReLateBound(ty::INNERMOST,
                                                                 ty::BrAnon(0))),
                                   param(0))], tcx.types.u64),

            "try" => {
                let mut_u8 = tcx.mk_mut_ptr(tcx.types.u8);
                let fn_ty = ty::Binder::bind(tcx.mk_fn_sig(
                    iter::once(mut_u8),
                    tcx.mk_unit(),
                    false,
                    hir::Unsafety::Normal,
                    Abi::Rust,
                ));
                (0, vec![tcx.mk_fn_ptr(fn_ty), mut_u8, mut_u8], tcx.types.i32)
            }

            "va_start" | "va_end" => {
                match mk_va_list_ty(hir::MutMutable) {
                    Some((va_list_ref_ty, _)) => (0, vec![va_list_ref_ty], tcx.mk_unit()),
                    None => bug!("`va_list` language item needed for C-variadic intrinsics")
                }
            }

            "va_copy" => {
                match mk_va_list_ty(hir::MutImmutable) {
                    Some((va_list_ref_ty, va_list_ty)) => {
                        let va_list_ptr_ty = tcx.mk_mut_ptr(va_list_ty);
                        (0, vec![va_list_ptr_ty, va_list_ref_ty], tcx.mk_unit())
                    }
                    None => bug!("`va_list` language item needed for C-variadic intrinsics")
                }
            }

            "va_arg" => {
                match mk_va_list_ty(hir::MutMutable) {
                    Some((va_list_ref_ty, _)) => (1, vec![va_list_ref_ty], param(0)),
                    None => bug!("`va_list` language item needed for C-variadic intrinsics")
                }
            }

            "nontemporal_store" => {
                (1, vec![ tcx.mk_mut_ptr(param(0)), param(0) ], tcx.mk_unit())
            }

            ref other => {
                struct_span_err!(tcx.sess, it.span, E0093,
                                 "unrecognized intrinsic function: `{}`",
                                 *other)
                                 .span_label(it.span, "unrecognized intrinsic")
                                 .emit();
                return;
            }
        };
        (n_tps, inputs, output, unsafety)
    };
    equate_intrinsic_type(tcx, it, n_tps, Abi::RustIntrinsic, unsafety, inputs, output)
}

/// Type-check `extern "platform-intrinsic" { ... }` functions.
pub fn check_platform_intrinsic_type(tcx: TyCtxt<'_>, it: &hir::ForeignItem) {
    let param = |n| {
        let name = InternedString::intern(&format!("P{}", n));
        tcx.mk_ty_param(n, name)
    };

    let name = it.ident.as_str();

    let (n_tps, inputs, output) = match &*name {
        "simd_eq" | "simd_ne" | "simd_lt" | "simd_le" | "simd_gt" | "simd_ge" => {
            (2, vec![param(0), param(0)], param(1))
        }
        "simd_add" | "simd_sub" | "simd_mul" | "simd_rem" |
        "simd_div" | "simd_shl" | "simd_shr" |
        "simd_and" | "simd_or" | "simd_xor" |
        "simd_fmin" | "simd_fmax" | "simd_fpow" |
        "simd_saturating_add" | "simd_saturating_sub" => {
            (1, vec![param(0), param(0)], param(0))
        }
        "simd_fsqrt" | "simd_fsin" | "simd_fcos" | "simd_fexp" | "simd_fexp2" |
        "simd_flog2" | "simd_flog10" | "simd_flog" |
        "simd_fabs" | "simd_floor" | "simd_ceil" => {
            (1, vec![param(0)], param(0))
        }
        "simd_fpowi" => {
            (1, vec![param(0), tcx.types.i32], param(0))
        }
        "simd_fma" => {
            (1, vec![param(0), param(0), param(0)], param(0))
        }
        "simd_gather" => {
            (3, vec![param(0), param(1), param(2)], param(0))
        }
        "simd_scatter" => {
            (3, vec![param(0), param(1), param(2)], tcx.mk_unit())
        }
        "simd_insert" => (2, vec![param(0), tcx.types.u32, param(1)], param(0)),
        "simd_extract" => (2, vec![param(0), tcx.types.u32], param(1)),
        "simd_cast" => (2, vec![param(0)], param(1)),
        "simd_bitmask" => (2, vec![param(0)], param(1)),
        "simd_select" |
        "simd_select_bitmask" => (2, vec![param(0), param(1), param(1)], param(1)),
        "simd_reduce_all" | "simd_reduce_any" => (1, vec![param(0)], tcx.types.bool),
        "simd_reduce_add_ordered" | "simd_reduce_mul_ordered"
            => (2, vec![param(0), param(1)], param(1)),
        "simd_reduce_add_unordered" | "simd_reduce_mul_unordered" |
        "simd_reduce_and" | "simd_reduce_or"  | "simd_reduce_xor" |
        "simd_reduce_min" | "simd_reduce_max" |
        "simd_reduce_min_nanless" | "simd_reduce_max_nanless"
            => (2, vec![param(0)], param(1)),
        name if name.starts_with("simd_shuffle") => {
            match name["simd_shuffle".len()..].parse() {
                Ok(n) => {
                    let params = vec![param(0), param(0),
                                      tcx.mk_array(tcx.types.u32, n)];
                    (2, params, param(1))
                }
                Err(_) => {
                    span_err!(tcx.sess, it.span, E0439,
                              "invalid `simd_shuffle`, needs length: `{}`", name);
                    return
                }
            }
        }
        _ => {
            let msg = format!("unrecognized platform-specific intrinsic function: `{}`", name);
            tcx.sess.span_err(it.span, &msg);
            return;
        }
    };

    equate_intrinsic_type(tcx, it, n_tps, Abi::PlatformIntrinsic, hir::Unsafety::Unsafe,
                          inputs, output)
}
