// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Type-checking for the rust-intrinsic and platform-intrinsic
//! intrinsics that the compiler exposes.

use astconv::AstConv;
use intrinsics;
use middle::subst;
use middle::ty::FnSig;
use middle::ty::{self, Ty};
use middle::ty::fold::TypeFolder;
use {CrateCtxt, require_same_types};

use std::collections::{HashMap};
use syntax::abi;
use syntax::ast;
use syntax::attr::AttrMetaMethods;
use syntax::codemap::Span;
use syntax::parse::token;

use rustc_front::hir;

fn equate_intrinsic_type<'a, 'tcx>(tcx: &ty::ctxt<'tcx>, it: &hir::ForeignItem,
                                   n_tps: usize,
                                   abi: abi::Abi,
                                   inputs: Vec<ty::Ty<'tcx>>,
                                   output: ty::FnOutput<'tcx>) {
    let fty = tcx.mk_fn(None, tcx.mk_bare_fn(ty::BareFnTy {
        unsafety: hir::Unsafety::Unsafe,
        abi: abi,
        sig: ty::Binder(FnSig {
            inputs: inputs,
            output: output,
            variadic: false,
        }),
    }));
    let i_ty = tcx.lookup_item_type(tcx.map.local_def_id(it.id));
    let i_n_tps = i_ty.generics.types.len(subst::FnSpace);
    if i_n_tps != n_tps {
        span_err!(tcx.sess, it.span, E0094,
            "intrinsic has wrong number of type \
             parameters: found {}, expected {}",
             i_n_tps, n_tps);
    } else {
        require_same_types(tcx,
                           None,
                           false,
                           it.span,
                           i_ty.ty,
                           fty,
                           || {
                format!("intrinsic has wrong type: expected `{}`",
                         fty)
            });
    }
}

/// Remember to add all intrinsics here, in librustc_trans/trans/intrinsic.rs,
/// and in libcore/intrinsics.rs
pub fn check_intrinsic_type(ccx: &CrateCtxt, it: &hir::ForeignItem) {
    fn param<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>, n: u32) -> Ty<'tcx> {
        let name = token::intern(&format!("P{}", n));
        ccx.tcx.mk_param(subst::FnSpace, n, name)
    }

    let tcx = ccx.tcx;
    let name = it.name.as_str();
    let (n_tps, inputs, output) = if name.starts_with("atomic_") {
        let split : Vec<&str> = name.split('_').collect();
        assert!(split.len() >= 2, "Atomic intrinsic not correct format");

        //We only care about the operation here
        let (n_tps, inputs, output) = match split[1] {
            "cxchg" => (1, vec!(tcx.mk_mut_ptr(param(ccx, 0)),
                                param(ccx, 0),
                                param(ccx, 0)),
                        param(ccx, 0)),
            "load" => (1, vec!(tcx.mk_imm_ptr(param(ccx, 0))),
                       param(ccx, 0)),
            "store" => (1, vec!(tcx.mk_mut_ptr(param(ccx, 0)), param(ccx, 0)),
                        tcx.mk_nil()),

            "xchg" | "xadd" | "xsub" | "and"  | "nand" | "or" | "xor" | "max" |
            "min"  | "umax" | "umin" => {
                (1, vec!(tcx.mk_mut_ptr(param(ccx, 0)), param(ccx, 0)),
                 param(ccx, 0))
            }
            "fence" | "singlethreadfence" => {
                (0, Vec::new(), tcx.mk_nil())
            }
            op => {
                span_err!(tcx.sess, it.span, E0092,
                    "unrecognized atomic operation function: `{}`", op);
                return;
            }
        };
        (n_tps, inputs, ty::FnConverging(output))
    } else if &name[..] == "abort" || &name[..] == "unreachable" {
        (0, Vec::new(), ty::FnDiverging)
    } else {
        let (n_tps, inputs, output) = match &name[..] {
            "breakpoint" => (0, Vec::new(), tcx.mk_nil()),
            "size_of" |
            "pref_align_of" | "min_align_of" => (1, Vec::new(), ccx.tcx.types.usize),
            "size_of_val" |  "min_align_of_val" => {
                (1, vec![
                    tcx.mk_imm_ref(tcx.mk_region(ty::ReLateBound(ty::DebruijnIndex::new(1),
                                                                  ty::BrAnon(0))),
                                    param(ccx, 0))
                 ], ccx.tcx.types.usize)
            }
            "init" | "init_dropped" => (1, Vec::new(), param(ccx, 0)),
            "uninit" => (1, Vec::new(), param(ccx, 0)),
            "forget" => (1, vec!( param(ccx, 0) ), tcx.mk_nil()),
            "transmute" => (2, vec!( param(ccx, 0) ), param(ccx, 1)),
            "move_val_init" => {
                (1,
                 vec!(
                    tcx.mk_mut_ptr(param(ccx, 0)),
                    param(ccx, 0)
                  ),
               tcx.mk_nil())
            }
            "drop_in_place" => {
                (1, vec![tcx.mk_mut_ptr(param(ccx, 0))], tcx.mk_nil())
            }
            "needs_drop" => (1, Vec::new(), ccx.tcx.types.bool),

            "type_name" => (1, Vec::new(), tcx.mk_static_str()),
            "type_id" => (1, Vec::new(), ccx.tcx.types.u64),
            "offset" | "arith_offset" => {
              (1,
               vec!(
                  tcx.mk_ptr(ty::TypeAndMut {
                      ty: param(ccx, 0),
                      mutbl: hir::MutImmutable
                  }),
                  ccx.tcx.types.isize
               ),
               tcx.mk_ptr(ty::TypeAndMut {
                   ty: param(ccx, 0),
                   mutbl: hir::MutImmutable
               }))
            }
            "copy" | "copy_nonoverlapping" => {
              (1,
               vec!(
                  tcx.mk_ptr(ty::TypeAndMut {
                      ty: param(ccx, 0),
                      mutbl: hir::MutImmutable
                  }),
                  tcx.mk_ptr(ty::TypeAndMut {
                      ty: param(ccx, 0),
                      mutbl: hir::MutMutable
                  }),
                  tcx.types.usize,
               ),
               tcx.mk_nil())
            }
            "volatile_copy_memory" | "volatile_copy_nonoverlapping_memory" => {
              (1,
               vec!(
                  tcx.mk_ptr(ty::TypeAndMut {
                      ty: param(ccx, 0),
                      mutbl: hir::MutMutable
                  }),
                  tcx.mk_ptr(ty::TypeAndMut {
                      ty: param(ccx, 0),
                      mutbl: hir::MutImmutable
                  }),
                  tcx.types.usize,
               ),
               tcx.mk_nil())
            }
            "write_bytes" | "volatile_set_memory" => {
              (1,
               vec!(
                  tcx.mk_ptr(ty::TypeAndMut {
                      ty: param(ccx, 0),
                      mutbl: hir::MutMutable
                  }),
                  tcx.types.u8,
                  tcx.types.usize,
               ),
               tcx.mk_nil())
            }
            "sqrt" | "sin" | "cos" | "exp" | "exp2" | "log" | "log10" | "log2" | "fabs" |
                "floor" | "ceil" | "trunc" | "rint" | "nearbyint" | "round" =>
                (1, vec!(param(ccx, 0)), param(ccx, 0)),

            "powi" =>
                (1, vec!(param(ccx, 0), tcx.types.i32), param(ccx, 0)),

            "pow" | "copysign" =>
                (1, vec!(param(ccx, 0), param(ccx, 0)), param(ccx, 0)),

            "fma" =>
                (1, vec!(param(ccx, 0), param(ccx, 0), param(ccx, 0)), param(ccx, 0)),

            "volatile_load" =>
                (1, vec!( tcx.mk_imm_ptr(param(ccx, 0)) ), param(ccx, 0)),
            "volatile_store" =>
                (1, vec!( tcx.mk_mut_ptr(param(ccx, 0)), param(ccx, 0) ), tcx.mk_nil()),

            "ctpop" | "ctlz" | "cttz" | "bswap" => (1, vec!(param(ccx, 0)), param(ccx, 0)),

            "add_with_overflow" | "sub_with_overflow"  | "mul_with_overflow" =>
                (1, vec!(param(ccx, 0), param(ccx, 0)),
                tcx.mk_tup(vec!(param(ccx, 0), tcx.types.bool))),

            "unchecked_div" | "unchecked_rem" =>
                (1, vec![param(ccx, 0), param(ccx, 0)], param(ccx, 0)),

            "overflowing_add" | "overflowing_sub" | "overflowing_mul" =>
                (1, vec![param(ccx, 0), param(ccx, 0)], param(ccx, 0)),

            "return_address" => (0, vec![], tcx.mk_imm_ptr(tcx.types.u8)),

            "assume" => (0, vec![tcx.types.bool], tcx.mk_nil()),

            "discriminant_value" => (1, vec![
                    tcx.mk_imm_ref(tcx.mk_region(ty::ReLateBound(ty::DebruijnIndex::new(1),
                                                                  ty::BrAnon(0))),
                                   param(ccx, 0))], tcx.types.u64),

            "try" => {
                let mut_u8 = tcx.mk_mut_ptr(tcx.types.u8);
                let fn_ty = ty::BareFnTy {
                    unsafety: hir::Unsafety::Normal,
                    abi: abi::Rust,
                    sig: ty::Binder(FnSig {
                        inputs: vec![mut_u8],
                        output: ty::FnOutput::FnConverging(tcx.mk_nil()),
                        variadic: false,
                    }),
                };
                let fn_ty = tcx.mk_bare_fn(fn_ty);
                (0, vec![tcx.mk_fn(None, fn_ty), mut_u8], mut_u8)
            }

            ref other => {
                span_err!(tcx.sess, it.span, E0093,
                          "unrecognized intrinsic function: `{}`", *other);
                return;
            }
        };
        (n_tps, inputs, ty::FnConverging(output))
    };
    equate_intrinsic_type(
        tcx,
        it,
        n_tps,
        abi::RustIntrinsic,
        inputs,
        output
        )
}

/// Type-check `extern "platform-intrinsic" { ... }` functions.
pub fn check_platform_intrinsic_type(ccx: &CrateCtxt,
                                     it: &hir::ForeignItem) {
    let param = |n| {
        let name = token::intern(&format!("P{}", n));
        ccx.tcx.mk_param(subst::FnSpace, n, name)
    };

    let tcx = ccx.tcx;
    let i_ty = tcx.lookup_item_type(tcx.map.local_def_id(it.id));
    let i_n_tps = i_ty.generics.types.len(subst::FnSpace);
    let name = it.name.as_str();

    let (n_tps, inputs, output) = match &*name {
        "simd_eq" | "simd_ne" | "simd_lt" | "simd_le" | "simd_gt" | "simd_ge" => {
            (2, vec![param(0), param(0)], param(1))
        }
        "simd_add" | "simd_sub" | "simd_mul" |
        "simd_div" | "simd_shl" | "simd_shr" |
        "simd_and" | "simd_or" | "simd_xor" => {
            (1, vec![param(0), param(0)], param(0))
        }
        "simd_insert" => (2, vec![param(0), tcx.types.u32, param(1)], param(0)),
        "simd_extract" => (2, vec![param(0), tcx.types.u32], param(1)),
        "simd_cast" => (2, vec![param(0)], param(1)),
        name if name.starts_with("simd_shuffle") => {
            match name["simd_shuffle".len()..].parse() {
                Ok(n) => {
                    let params = vec![param(0), param(0),
                                      tcx.mk_ty(ty::TyArray(tcx.types.u32, n))];
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
            match intrinsics::Intrinsic::find(tcx, &name) {
                Some(intr) => {
                    // this function is a platform specific intrinsic
                    if i_n_tps != 0 {
                        span_err!(tcx.sess, it.span, E0440,
                                  "platform-specific intrinsic has wrong number of type \
                                   parameters: found {}, expected 0",
                                  i_n_tps);
                        return
                    }

                    let mut structural_to_nomimal = HashMap::new();

                    let sig = tcx.no_late_bound_regions(i_ty.ty.fn_sig()).unwrap();
                    if intr.inputs.len() != sig.inputs.len() {
                        span_err!(tcx.sess, it.span, E0444,
                                  "platform-specific intrinsic has invalid number of \
                                   arguments: found {}, expected {}",
                                  intr.inputs.len(), sig.inputs.len());
                        return
                    }
                    let input_pairs = intr.inputs.iter().zip(&sig.inputs);
                    for (i, (expected_arg, arg)) in input_pairs.enumerate() {
                        match_intrinsic_type_to_type(tcx, &format!("argument {}", i + 1), it.span,
                                                     &mut structural_to_nomimal, expected_arg, arg);
                    }
                    match_intrinsic_type_to_type(tcx, "return value", it.span,
                                                 &mut structural_to_nomimal,
                                                 &intr.output, sig.output.unwrap());
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

    equate_intrinsic_type(
        tcx,
        it,
        n_tps,
        abi::PlatformIntrinsic,
        inputs,
        ty::FnConverging(output)
        )
}

// walk the expected type and the actual type in lock step, checking they're
// the same, in a kinda-structural way, i.e. `Vector`s have to be simd structs with
// exactly the right element type
fn match_intrinsic_type_to_type<'tcx, 'a>(
        tcx: &ty::ctxt<'tcx>,
        position: &str,
        span: Span,
        structural_to_nominal: &mut HashMap<&'a intrinsics::Type, ty::Ty<'tcx>>,
        expected: &'a intrinsics::Type, t: ty::Ty<'tcx>)
{
    use intrinsics::Type::*;

    let simple_error = |real: &str, expected: &str| {
        span_err!(tcx.sess, span, E0442,
                  "intrinsic {} has wrong type: found {}, expected {}",
                  position, real, expected)
    };

    match *expected {
        Void => match t.sty {
            ty::TyTuple(ref v) if v.is_empty() => {},
            _ => simple_error(&format!("`{}`", t), "()"),
        },
        // (The width we pass to LLVM doesn't concern the type checker.)
        Integer(signed, bits, _llvm_width) => match (signed, bits, &t.sty) {
            (true,  8,  &ty::TyInt(ast::IntTy::TyI8)) |
            (false, 8,  &ty::TyUint(ast::UintTy::TyU8)) |
            (true,  16, &ty::TyInt(ast::IntTy::TyI16)) |
            (false, 16, &ty::TyUint(ast::UintTy::TyU16)) |
            (true,  32, &ty::TyInt(ast::IntTy::TyI32)) |
            (false, 32, &ty::TyUint(ast::UintTy::TyU32)) |
            (true,  64, &ty::TyInt(ast::IntTy::TyI64)) |
            (false, 64, &ty::TyUint(ast::UintTy::TyU64)) => {},
            _ => simple_error(&format!("`{}`", t),
                              &format!("`{}{n}`",
                                       if signed {"i"} else {"u"},
                                       n = bits)),
        },
        Float(bits) => match (bits, &t.sty) {
            (32, &ty::TyFloat(ast::FloatTy::TyF32)) |
            (64, &ty::TyFloat(ast::FloatTy::TyF64)) => {},
            _ => simple_error(&format!("`{}`", t),
                              &format!("`f{n}`", n = bits)),
        },
        Pointer(ref inner_expected, ref _llvm_type, const_) => {
            match t.sty {
                ty::TyRawPtr(ty::TypeAndMut { ty, mutbl }) => {
                    if (mutbl == hir::MutImmutable) != const_ {
                        simple_error(&format!("`{}`", t),
                                     if const_ {"const pointer"} else {"mut pointer"})
                    }
                    match_intrinsic_type_to_type(tcx, position, span, structural_to_nominal,
                                                 inner_expected, ty)
                }
                _ => simple_error(&format!("`{}`", t),
                                  &format!("raw pointer")),
            }
        }
        Vector(ref inner_expected, ref _llvm_type, len) => {
            if !t.is_simd() {
                simple_error(&format!("non-simd type `{}`", t),
                             "simd type");
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
                ty::TyTuple(ref contents) => {
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
                                  &format!("tuple")),
            }
        }
    }
}
