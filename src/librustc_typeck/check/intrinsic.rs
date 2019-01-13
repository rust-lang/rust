//! Type-checking for the rust-intrinsic and platform-intrinsic
//! intrinsics that the compiler exposes.

use intrinsics;
use rustc::traits::{ObligationCause, ObligationCauseCode};
use rustc::ty::{self, TyCtxt, Ty};
use rustc::ty::subst::Subst;
use rustc::util::nodemap::FxHashMap;
use require_same_types;

use rustc_target::spec::abi::Abi;
use syntax::ast;
use syntax::symbol::Symbol;
use syntax_pos::Span;

use rustc::hir;

use std::iter;

fn equate_intrinsic_type<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    it: &hir::ForeignItem,
    n_tps: usize,
    abi: Abi,
    safety: hir::Unsafety,
    inputs: Vec<Ty<'tcx>>,
    output: Ty<'tcx>,
) {
    let def_id = tcx.hir().local_def_id(it.id);

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
    let cause = ObligationCause::new(it.span, it.id, ObligationCauseCode::IntrinsicType);
    require_same_types(tcx, &cause, tcx.mk_fn_ptr(tcx.fn_sig(def_id)), fty);
}

/// Returns whether the given intrinsic is unsafe to call or not.
pub fn intrisic_operation_unsafety(intrinsic: &str) -> hir::Unsafety {
    match intrinsic {
        "size_of" | "min_align_of" | "needs_drop" |
        "add_with_overflow" | "sub_with_overflow" | "mul_with_overflow" |
        "overflowing_add" | "overflowing_sub" | "overflowing_mul" |
        "rotate_left" | "rotate_right" |
        "ctpop" | "ctlz" | "cttz" | "bswap" | "bitreverse"
        => hir::Unsafety::Normal,
        _ => hir::Unsafety::Unsafe,
    }
}

/// Remember to add all intrinsics here, in librustc_codegen_llvm/intrinsic.rs,
/// and in libcore/intrinsics.rs
pub fn check_intrinsic_type<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                      it: &hir::ForeignItem) {
    let param = |n| tcx.mk_ty_param(n, Symbol::intern(&format!("P{}", n)).as_interned_str());
    let name = it.ident.as_str();

    let mk_va_list_ty = || {
        tcx.lang_items().va_list().map(|did| {
            let region = tcx.mk_region(ty::ReLateBound(ty::INNERMOST, ty::BrAnon(0)));
            let env_region = ty::ReLateBound(ty::INNERMOST, ty::BrEnv);
            let va_list_ty = tcx.type_of(did).subst(tcx, &[region.into()]);
            tcx.mk_mut_ref(tcx.mk_region(env_region), va_list_ty)
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
            "uninit" => (1, Vec::new(), param(0)),
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

            "overflowing_add" | "overflowing_sub" | "overflowing_mul" =>
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
                match mk_va_list_ty() {
                    Some(va_list_ty) => (0, vec![va_list_ty], tcx.mk_unit()),
                    None => bug!("va_list lang_item must be defined to use va_list intrinsics")
                }
            }

            "va_copy" => {
                match tcx.lang_items().va_list() {
                    Some(did) => {
                        let region = tcx.mk_region(ty::ReLateBound(ty::INNERMOST, ty::BrAnon(0)));
                        let env_region = ty::ReLateBound(ty::INNERMOST, ty::BrEnv);
                        let va_list_ty = tcx.type_of(did).subst(tcx, &[region.into()]);
                        let ret_ty = match va_list_ty.sty {
                            ty::Adt(def, _) if def.is_struct() => {
                                let fields = &def.non_enum_variant().fields;
                                match tcx.type_of(fields[0].did).subst(tcx, &[region.into()]).sty {
                                    ty::Ref(_, element_ty, _) => match element_ty.sty {
                                        ty::Adt(..) => element_ty,
                                        _ => va_list_ty
                                    }
                                    _ => bug!("va_list structure is invalid")
                                }
                            }
                            _ => {
                                bug!("va_list structure is invalid")
                            }
                        };
                        (0, vec![tcx.mk_imm_ref(tcx.mk_region(env_region), va_list_ty)], ret_ty)
                    }
                    None => bug!("va_list lang_item must be defined to use va_list intrinsics")
                }
            }

            "va_arg" => {
                match mk_va_list_ty() {
                    Some(va_list_ty) => (1, vec![va_list_ty], param(0)),
                    None => bug!("va_list lang_item must be defined to use va_list intrinsics")
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
pub fn check_platform_intrinsic_type<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                               it: &hir::ForeignItem) {
    let param = |n| {
        let name = Symbol::intern(&format!("P{}", n)).as_interned_str();
        tcx.mk_ty_param(n, name)
    };

    let def_id = tcx.hir().local_def_id(it.id);
    let i_n_tps = tcx.generics_of(def_id).own_counts().types;
    let name = it.ident.as_str();

    let (n_tps, inputs, output) = match &*name {
        "simd_eq" | "simd_ne" | "simd_lt" | "simd_le" | "simd_gt" | "simd_ge" => {
            (2, vec![param(0), param(0)], param(1))
        }
        "simd_add" | "simd_sub" | "simd_mul" | "simd_rem" |
        "simd_div" | "simd_shl" | "simd_shr" |
        "simd_and" | "simd_or" | "simd_xor" |
        "simd_fmin" | "simd_fmax" | "simd_fpow" => {
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
            match intrinsics::Intrinsic::find(&name) {
                Some(intr) => {
                    // this function is a platform specific intrinsic
                    if i_n_tps != 0 {
                        span_err!(tcx.sess, it.span, E0440,
                                  "platform-specific intrinsic has wrong number of type \
                                   parameters: found {}, expected 0",
                                  i_n_tps);
                        return
                    }

                    let mut structural_to_nomimal = FxHashMap::default();

                    let sig = tcx.fn_sig(def_id);
                    let sig = sig.no_bound_vars().unwrap();
                    if intr.inputs.len() != sig.inputs().len() {
                        span_err!(tcx.sess, it.span, E0444,
                                  "platform-specific intrinsic has invalid number of \
                                   arguments: found {}, expected {}",
                                  sig.inputs().len(), intr.inputs.len());
                        return
                    }
                    let input_pairs = intr.inputs.iter().zip(sig.inputs());
                    for (i, (expected_arg, arg)) in input_pairs.enumerate() {
                        match_intrinsic_type_to_type(tcx, &format!("argument {}", i + 1), it.span,
                                                     &mut structural_to_nomimal, expected_arg, arg);
                    }
                    match_intrinsic_type_to_type(tcx, "return value", it.span,
                                                 &mut structural_to_nomimal,
                                                 &intr.output, sig.output());
                    return
                }
                None => {
                    span_err!(tcx.sess, it.span, E0441,
                              "unrecognized platform-specific intrinsic function: `{}`", name);
                    return;
                }
            }
        }
    };

    equate_intrinsic_type(tcx, it, n_tps, Abi::PlatformIntrinsic, hir::Unsafety::Unsafe,
                          inputs, output)
}

// walk the expected type and the actual type in lock step, checking they're
// the same, in a kinda-structural way, i.e., `Vector`s have to be simd structs with
// exactly the right element type
fn match_intrinsic_type_to_type<'a, 'tcx>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        position: &str,
        span: Span,
        structural_to_nominal: &mut FxHashMap<&'a intrinsics::Type, Ty<'tcx>>,
        expected: &'a intrinsics::Type, t: Ty<'tcx>)
{
    use intrinsics::Type::*;

    let simple_error = |real: &str, expected: &str| {
        span_err!(tcx.sess, span, E0442,
                  "intrinsic {} has wrong type: found {}, expected {}",
                  position, real, expected)
    };

    match *expected {
        Void => match t.sty {
            ty::Tuple(ref v) if v.is_empty() => {},
            _ => simple_error(&format!("`{}`", t), "()"),
        },
        // (The width we pass to LLVM doesn't concern the type checker.)
        Integer(signed, bits, _llvm_width) => match (signed, bits, &t.sty) {
            (true,  8,  &ty::Int(ast::IntTy::I8)) |
            (false, 8,  &ty::Uint(ast::UintTy::U8)) |
            (true,  16, &ty::Int(ast::IntTy::I16)) |
            (false, 16, &ty::Uint(ast::UintTy::U16)) |
            (true,  32, &ty::Int(ast::IntTy::I32)) |
            (false, 32, &ty::Uint(ast::UintTy::U32)) |
            (true,  64, &ty::Int(ast::IntTy::I64)) |
            (false, 64, &ty::Uint(ast::UintTy::U64)) |
            (true,  128, &ty::Int(ast::IntTy::I128)) |
            (false, 128, &ty::Uint(ast::UintTy::U128)) => {},
            _ => simple_error(&format!("`{}`", t),
                              &format!("`{}{n}`",
                                       if signed {"i"} else {"u"},
                                       n = bits)),
        },
        Float(bits) => match (bits, &t.sty) {
            (32, &ty::Float(ast::FloatTy::F32)) |
            (64, &ty::Float(ast::FloatTy::F64)) => {},
            _ => simple_error(&format!("`{}`", t),
                              &format!("`f{n}`", n = bits)),
        },
        Pointer(ref inner_expected, ref _llvm_type, const_) => {
            match t.sty {
                ty::RawPtr(ty::TypeAndMut { ty, mutbl }) => {
                    if (mutbl == hir::MutImmutable) != const_ {
                        simple_error(&format!("`{}`", t),
                                     if const_ {"const pointer"} else {"mut pointer"})
                    }
                    match_intrinsic_type_to_type(tcx, position, span, structural_to_nominal,
                                                 inner_expected, ty)
                }
                _ => simple_error(&format!("`{}`", t), "raw pointer"),
            }
        }
        Vector(ref inner_expected, ref _llvm_type, len) => {
            if !t.is_simd() {
                simple_error(&format!("non-simd type `{}`", t), "simd type");
                return;
            }
            let t_len = t.simd_size(tcx);
            if len as usize != t_len {
                simple_error(&format!("vector with length {}", t_len),
                             &format!("length {}", len));
                return;
            }
            let t_ty = t.simd_type(tcx);
            {
                // check that a given structural type always has the same an intrinsic definition
                let previous = structural_to_nominal.entry(expected).or_insert(t);
                if *previous != t {
                    // this gets its own error code because it is non-trivial
                    span_err!(tcx.sess, span, E0443,
                              "intrinsic {} has wrong type: found `{}`, expected `{}` which \
                               was used for this vector type previously in this signature",
                              position,
                              t,
                              *previous);
                    return;
                }
            }
            match_intrinsic_type_to_type(tcx,
                                         position,
                                         span,
                                         structural_to_nominal,
                                         inner_expected,
                                         t_ty)
        }
        Aggregate(_flatten, ref expected_contents) => {
            match t.sty {
                ty::Tuple(contents) => {
                    if contents.len() != expected_contents.len() {
                        simple_error(&format!("tuple with length {}", contents.len()),
                                     &format!("tuple with length {}", expected_contents.len()));
                        return
                    }
                    for (e, c) in expected_contents.iter().zip(contents) {
                        match_intrinsic_type_to_type(tcx, position, span, structural_to_nominal,
                                                     e, c)
                    }
                }
                _ => simple_error(&format!("`{}`", t),
                                  "tuple"),
            }
        }
    }
}
