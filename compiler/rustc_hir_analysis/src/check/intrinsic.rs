//! Type-checking for the `#[rustc_intrinsic]` intrinsics that the compiler exposes.

use rustc_abi::ExternAbi;
use rustc_errors::DiagMessage;
use rustc_hir::{self as hir, LangItem};
use rustc_middle::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, Symbol, sym};

use crate::check::check_function_signature;
use crate::errors::{
    UnrecognizedAtomicOperation, UnrecognizedIntrinsicFunction,
    WrongNumberOfGenericArgumentsToIntrinsic,
};

fn equate_intrinsic_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    span: Span,
    def_id: LocalDefId,
    n_tps: usize,
    n_lts: usize,
    n_cts: usize,
    sig: ty::PolyFnSig<'tcx>,
) {
    let (generics, span) = match tcx.hir_node_by_def_id(def_id) {
        hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn { generics, .. }, .. }) => {
            (tcx.generics_of(def_id), generics.span)
        }
        _ => tcx.dcx().span_bug(span, "intrinsic must be a function"),
    };
    let own_counts = generics.own_counts();

    let gen_count_ok = |found: usize, expected: usize, descr: &str| -> bool {
        if found != expected {
            tcx.dcx().emit_err(WrongNumberOfGenericArgumentsToIntrinsic {
                span,
                found,
                expected,
                descr,
            });
            false
        } else {
            true
        }
    };

    // the host effect param should be invisible as it shouldn't matter
    // whether effects is enabled for the intrinsic provider crate.
    if gen_count_ok(own_counts.lifetimes, n_lts, "lifetime")
        && gen_count_ok(own_counts.types, n_tps, "type")
        && gen_count_ok(own_counts.consts, n_cts, "const")
    {
        let _ = check_function_signature(
            tcx,
            ObligationCause::new(span, def_id, ObligationCauseCode::IntrinsicType),
            def_id.into(),
            sig,
        );
    }
}

/// Returns the unsafety of the given intrinsic.
fn intrinsic_operation_unsafety(tcx: TyCtxt<'_>, intrinsic_id: LocalDefId) -> hir::Safety {
    let is_in_list = match tcx.item_name(intrinsic_id.into()) {
        // When adding a new intrinsic to this list,
        // it's usually worth updating that intrinsic's documentation
        // to note that it's safe to call, since
        // safe extern fns are otherwise unprecedented.
        sym::abort
        | sym::assert_inhabited
        | sym::assert_zero_valid
        | sym::assert_mem_uninitialized_valid
        | sym::box_new
        | sym::breakpoint
        | sym::size_of
        | sym::min_align_of
        | sym::needs_drop
        | sym::caller_location
        | sym::add_with_overflow
        | sym::sub_with_overflow
        | sym::mul_with_overflow
        | sym::carrying_mul_add
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
        | sym::three_way_compare
        | sym::discriminant_value
        | sym::type_id
        | sym::select_unpredictable
        | sym::cold_path
        | sym::ptr_guaranteed_cmp
        | sym::minnumf16
        | sym::minnumf32
        | sym::minnumf64
        | sym::minnumf128
        | sym::maxnumf16
        | sym::maxnumf32
        | sym::maxnumf64
        | sym::maxnumf128
        | sym::rustc_peek
        | sym::type_name
        | sym::forget
        | sym::black_box
        | sym::variant_count
        | sym::is_val_statically_known
        | sym::ptr_mask
        | sym::aggregate_raw_ptr
        | sym::ptr_metadata
        | sym::ub_checks
        | sym::contract_checks
        | sym::contract_check_requires
        | sym::contract_check_ensures
        | sym::fadd_algebraic
        | sym::fsub_algebraic
        | sym::fmul_algebraic
        | sym::fdiv_algebraic
        | sym::frem_algebraic
        | sym::round_ties_even_f16
        | sym::round_ties_even_f32
        | sym::round_ties_even_f64
        | sym::round_ties_even_f128
        | sym::const_eval_select => hir::Safety::Safe,
        _ => hir::Safety::Unsafe,
    };

    if tcx.fn_sig(intrinsic_id).skip_binder().safety() != is_in_list {
        tcx.dcx().struct_span_err(
            tcx.def_span(intrinsic_id),
            DiagMessage::from(format!(
                "intrinsic safety mismatch between list of intrinsics within the compiler and core library intrinsics for intrinsic `{}`",
                tcx.item_name(intrinsic_id.into())
            )
        )).emit();
    }

    is_in_list
}

/// Remember to add all intrinsics here, in `compiler/rustc_codegen_llvm/src/intrinsic.rs`,
/// and in `library/core/src/intrinsics.rs`.
pub(crate) fn check_intrinsic_type(
    tcx: TyCtxt<'_>,
    intrinsic_id: LocalDefId,
    span: Span,
    intrinsic_name: Symbol,
) {
    let generics = tcx.generics_of(intrinsic_id);
    let param = |n| {
        if let &ty::GenericParamDef { name, kind: ty::GenericParamDefKind::Type { .. }, .. } =
            generics.param_at(n as usize, tcx)
        {
            Ty::new_param(tcx, n, name)
        } else {
            Ty::new_error_with_message(tcx, span, "expected param")
        }
    };
    let name_str = intrinsic_name.as_str();

    let bound_vars = tcx.mk_bound_variable_kinds(&[
        ty::BoundVariableKind::Region(ty::BoundRegionKind::Anon),
        ty::BoundVariableKind::Region(ty::BoundRegionKind::Anon),
        ty::BoundVariableKind::Region(ty::BoundRegionKind::ClosureEnv),
    ]);
    let mk_va_list_ty = |mutbl| {
        let did = tcx.require_lang_item(LangItem::VaList, Some(span));
        let region = ty::Region::new_bound(
            tcx,
            ty::INNERMOST,
            ty::BoundRegion { var: ty::BoundVar::ZERO, kind: ty::BoundRegionKind::Anon },
        );
        let env_region = ty::Region::new_bound(
            tcx,
            ty::INNERMOST,
            ty::BoundRegion {
                var: ty::BoundVar::from_u32(2),
                kind: ty::BoundRegionKind::ClosureEnv,
            },
        );
        let va_list_ty = tcx.type_of(did).instantiate(tcx, &[region.into()]);
        (Ty::new_ref(tcx, env_region, va_list_ty, mutbl), va_list_ty)
    };

    let (n_tps, n_lts, n_cts, inputs, output, safety) = if name_str.starts_with("atomic_") {
        let split: Vec<&str> = name_str.split('_').collect();
        assert!(split.len() >= 2, "Atomic intrinsic in an incorrect format");

        // Each atomic op has variants with different suffixes (`_seq_cst`, `_acquire`, etc.). Use
        // string ops to strip the suffixes, because the variants all get the same treatment here.
        let (n_tps, inputs, output) = match split[1] {
            "cxchg" | "cxchgweak" => (
                1,
                vec![Ty::new_mut_ptr(tcx, param(0)), param(0), param(0)],
                Ty::new_tup(tcx, &[param(0), tcx.types.bool]),
            ),
            "load" => (1, vec![Ty::new_imm_ptr(tcx, param(0))], param(0)),
            "store" => (1, vec![Ty::new_mut_ptr(tcx, param(0)), param(0)], tcx.types.unit),

            "xchg" | "xadd" | "xsub" | "and" | "nand" | "or" | "xor" | "max" | "min" | "umax"
            | "umin" => (1, vec![Ty::new_mut_ptr(tcx, param(0)), param(0)], param(0)),
            "fence" | "singlethreadfence" => (0, Vec::new(), tcx.types.unit),
            op => {
                tcx.dcx().emit_err(UnrecognizedAtomicOperation { span, op });
                return;
            }
        };
        (n_tps, 0, 0, inputs, output, hir::Safety::Unsafe)
    } else if intrinsic_name == sym::contract_check_ensures {
        // contract_check_ensures::<Ret, C>(Ret, C) -> Ret
        // where C: for<'a> Fn(&'a Ret) -> bool,
        //
        // so: two type params, 0 lifetime param, 0 const params, two inputs, no return
        (2, 0, 0, vec![param(0), param(1)], param(1), hir::Safety::Safe)
    } else {
        let safety = intrinsic_operation_unsafety(tcx, intrinsic_id);
        let (n_tps, n_cts, inputs, output) = match intrinsic_name {
            sym::abort => (0, 0, vec![], tcx.types.never),
            sym::unreachable => (0, 0, vec![], tcx.types.never),
            sym::breakpoint => (0, 0, vec![], tcx.types.unit),
            sym::size_of | sym::pref_align_of | sym::min_align_of | sym::variant_count => {
                (1, 0, vec![], tcx.types.usize)
            }
            sym::size_of_val | sym::min_align_of_val => {
                (1, 0, vec![Ty::new_imm_ptr(tcx, param(0))], tcx.types.usize)
            }
            sym::rustc_peek => (1, 0, vec![param(0)], param(0)),
            sym::caller_location => (0, 0, vec![], tcx.caller_location_ty()),
            sym::assert_inhabited
            | sym::assert_zero_valid
            | sym::assert_mem_uninitialized_valid => (1, 0, vec![], tcx.types.unit),
            sym::forget => (1, 0, vec![param(0)], tcx.types.unit),
            sym::transmute | sym::transmute_unchecked => (2, 0, vec![param(0)], param(1)),
            sym::prefetch_read_data
            | sym::prefetch_write_data
            | sym::prefetch_read_instruction
            | sym::prefetch_write_instruction => {
                (1, 0, vec![Ty::new_imm_ptr(tcx, param(0)), tcx.types.i32], tcx.types.unit)
            }
            sym::needs_drop => (1, 0, vec![], tcx.types.bool),

            sym::type_name => (1, 0, vec![], Ty::new_static_str(tcx)),
            sym::type_id => (1, 0, vec![], tcx.types.u128),
            sym::offset => (2, 0, vec![param(0), param(1)], param(0)),
            sym::arith_offset => (
                1,
                0,
                vec![Ty::new_imm_ptr(tcx, param(0)), tcx.types.isize],
                Ty::new_imm_ptr(tcx, param(0)),
            ),
            sym::ptr_mask => (
                1,
                0,
                vec![Ty::new_imm_ptr(tcx, param(0)), tcx.types.usize],
                Ty::new_imm_ptr(tcx, param(0)),
            ),

            sym::copy | sym::copy_nonoverlapping => (
                1,
                0,
                vec![
                    Ty::new_imm_ptr(tcx, param(0)),
                    Ty::new_mut_ptr(tcx, param(0)),
                    tcx.types.usize,
                ],
                tcx.types.unit,
            ),
            sym::volatile_copy_memory | sym::volatile_copy_nonoverlapping_memory => (
                1,
                0,
                vec![
                    Ty::new_mut_ptr(tcx, param(0)),
                    Ty::new_imm_ptr(tcx, param(0)),
                    tcx.types.usize,
                ],
                tcx.types.unit,
            ),
            sym::compare_bytes => {
                let byte_ptr = Ty::new_imm_ptr(tcx, tcx.types.u8);
                (0, 0, vec![byte_ptr, byte_ptr, tcx.types.usize], tcx.types.i32)
            }
            sym::write_bytes | sym::volatile_set_memory => (
                1,
                0,
                vec![Ty::new_mut_ptr(tcx, param(0)), tcx.types.u8, tcx.types.usize],
                tcx.types.unit,
            ),

            sym::sqrtf16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::sqrtf32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::sqrtf64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::sqrtf128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::powif16 => (0, 0, vec![tcx.types.f16, tcx.types.i32], tcx.types.f16),
            sym::powif32 => (0, 0, vec![tcx.types.f32, tcx.types.i32], tcx.types.f32),
            sym::powif64 => (0, 0, vec![tcx.types.f64, tcx.types.i32], tcx.types.f64),
            sym::powif128 => (0, 0, vec![tcx.types.f128, tcx.types.i32], tcx.types.f128),

            sym::sinf16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::sinf32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::sinf64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::sinf128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::cosf16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::cosf32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::cosf64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::cosf128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::powf16 => (0, 0, vec![tcx.types.f16, tcx.types.f16], tcx.types.f16),
            sym::powf32 => (0, 0, vec![tcx.types.f32, tcx.types.f32], tcx.types.f32),
            sym::powf64 => (0, 0, vec![tcx.types.f64, tcx.types.f64], tcx.types.f64),
            sym::powf128 => (0, 0, vec![tcx.types.f128, tcx.types.f128], tcx.types.f128),

            sym::expf16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::expf32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::expf64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::expf128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::exp2f16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::exp2f32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::exp2f64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::exp2f128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::logf16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::logf32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::logf64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::logf128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::log10f16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::log10f32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::log10f64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::log10f128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::log2f16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::log2f32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::log2f64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::log2f128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::fmaf16 => (0, 0, vec![tcx.types.f16, tcx.types.f16, tcx.types.f16], tcx.types.f16),
            sym::fmaf32 => (0, 0, vec![tcx.types.f32, tcx.types.f32, tcx.types.f32], tcx.types.f32),
            sym::fmaf64 => (0, 0, vec![tcx.types.f64, tcx.types.f64, tcx.types.f64], tcx.types.f64),
            sym::fmaf128 => {
                (0, 0, vec![tcx.types.f128, tcx.types.f128, tcx.types.f128], tcx.types.f128)
            }

            sym::fmuladdf16 => {
                (0, 0, vec![tcx.types.f16, tcx.types.f16, tcx.types.f16], tcx.types.f16)
            }
            sym::fmuladdf32 => {
                (0, 0, vec![tcx.types.f32, tcx.types.f32, tcx.types.f32], tcx.types.f32)
            }
            sym::fmuladdf64 => {
                (0, 0, vec![tcx.types.f64, tcx.types.f64, tcx.types.f64], tcx.types.f64)
            }
            sym::fmuladdf128 => {
                (0, 0, vec![tcx.types.f128, tcx.types.f128, tcx.types.f128], tcx.types.f128)
            }

            sym::fabsf16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::fabsf32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::fabsf64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::fabsf128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::minnumf16 => (0, 0, vec![tcx.types.f16, tcx.types.f16], tcx.types.f16),
            sym::minnumf32 => (0, 0, vec![tcx.types.f32, tcx.types.f32], tcx.types.f32),
            sym::minnumf64 => (0, 0, vec![tcx.types.f64, tcx.types.f64], tcx.types.f64),
            sym::minnumf128 => (0, 0, vec![tcx.types.f128, tcx.types.f128], tcx.types.f128),

            sym::maxnumf16 => (0, 0, vec![tcx.types.f16, tcx.types.f16], tcx.types.f16),
            sym::maxnumf32 => (0, 0, vec![tcx.types.f32, tcx.types.f32], tcx.types.f32),
            sym::maxnumf64 => (0, 0, vec![tcx.types.f64, tcx.types.f64], tcx.types.f64),
            sym::maxnumf128 => (0, 0, vec![tcx.types.f128, tcx.types.f128], tcx.types.f128),

            sym::copysignf16 => (0, 0, vec![tcx.types.f16, tcx.types.f16], tcx.types.f16),
            sym::copysignf32 => (0, 0, vec![tcx.types.f32, tcx.types.f32], tcx.types.f32),
            sym::copysignf64 => (0, 0, vec![tcx.types.f64, tcx.types.f64], tcx.types.f64),
            sym::copysignf128 => (0, 0, vec![tcx.types.f128, tcx.types.f128], tcx.types.f128),

            sym::floorf16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::floorf32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::floorf64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::floorf128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::ceilf16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::ceilf32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::ceilf64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::ceilf128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::truncf16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::truncf32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::truncf64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::truncf128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::round_ties_even_f16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::round_ties_even_f32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::round_ties_even_f64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::round_ties_even_f128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::roundf16 => (0, 0, vec![tcx.types.f16], tcx.types.f16),
            sym::roundf32 => (0, 0, vec![tcx.types.f32], tcx.types.f32),
            sym::roundf64 => (0, 0, vec![tcx.types.f64], tcx.types.f64),
            sym::roundf128 => (0, 0, vec![tcx.types.f128], tcx.types.f128),

            sym::volatile_load | sym::unaligned_volatile_load => {
                (1, 0, vec![Ty::new_imm_ptr(tcx, param(0))], param(0))
            }
            sym::volatile_store | sym::unaligned_volatile_store => {
                (1, 0, vec![Ty::new_mut_ptr(tcx, param(0)), param(0)], tcx.types.unit)
            }

            sym::ctpop | sym::ctlz | sym::ctlz_nonzero | sym::cttz | sym::cttz_nonzero => {
                (1, 0, vec![param(0)], tcx.types.u32)
            }

            sym::bswap | sym::bitreverse => (1, 0, vec![param(0)], param(0)),

            sym::three_way_compare => {
                (1, 0, vec![param(0), param(0)], tcx.ty_ordering_enum(Some(span)))
            }

            sym::add_with_overflow | sym::sub_with_overflow | sym::mul_with_overflow => {
                (1, 0, vec![param(0), param(0)], Ty::new_tup(tcx, &[param(0), tcx.types.bool]))
            }

            sym::carrying_mul_add => {
                (2, 0, vec![param(0); 4], Ty::new_tup(tcx, &[param(1), param(0)]))
            }

            sym::ptr_guaranteed_cmp => (
                1,
                0,
                vec![Ty::new_imm_ptr(tcx, param(0)), Ty::new_imm_ptr(tcx, param(0))],
                tcx.types.u8,
            ),

            sym::const_allocate => {
                (0, 0, vec![tcx.types.usize, tcx.types.usize], Ty::new_mut_ptr(tcx, tcx.types.u8))
            }
            sym::const_deallocate => (
                0,
                0,
                vec![Ty::new_mut_ptr(tcx, tcx.types.u8), tcx.types.usize, tcx.types.usize],
                tcx.types.unit,
            ),

            sym::ptr_offset_from => (
                1,
                0,
                vec![Ty::new_imm_ptr(tcx, param(0)), Ty::new_imm_ptr(tcx, param(0))],
                tcx.types.isize,
            ),
            sym::ptr_offset_from_unsigned => (
                1,
                0,
                vec![Ty::new_imm_ptr(tcx, param(0)), Ty::new_imm_ptr(tcx, param(0))],
                tcx.types.usize,
            ),
            sym::unchecked_div | sym::unchecked_rem | sym::exact_div | sym::disjoint_bitor => {
                (1, 0, vec![param(0), param(0)], param(0))
            }
            sym::unchecked_shl | sym::unchecked_shr => (2, 0, vec![param(0), param(1)], param(0)),
            sym::rotate_left | sym::rotate_right => (1, 0, vec![param(0), tcx.types.u32], param(0)),
            sym::unchecked_add | sym::unchecked_sub | sym::unchecked_mul => {
                (1, 0, vec![param(0), param(0)], param(0))
            }
            sym::wrapping_add | sym::wrapping_sub | sym::wrapping_mul => {
                (1, 0, vec![param(0), param(0)], param(0))
            }
            sym::saturating_add | sym::saturating_sub => (1, 0, vec![param(0), param(0)], param(0)),
            sym::fadd_fast | sym::fsub_fast | sym::fmul_fast | sym::fdiv_fast | sym::frem_fast => {
                (1, 0, vec![param(0), param(0)], param(0))
            }
            sym::fadd_algebraic
            | sym::fsub_algebraic
            | sym::fmul_algebraic
            | sym::fdiv_algebraic
            | sym::frem_algebraic => (1, 0, vec![param(0), param(0)], param(0)),
            sym::float_to_int_unchecked => (2, 0, vec![param(0)], param(1)),

            sym::assume => (0, 0, vec![tcx.types.bool], tcx.types.unit),
            sym::select_unpredictable => (1, 0, vec![tcx.types.bool, param(0), param(0)], param(0)),
            sym::cold_path => (0, 0, vec![], tcx.types.unit),

            sym::read_via_copy => (1, 0, vec![Ty::new_imm_ptr(tcx, param(0))], param(0)),
            sym::write_via_move => {
                (1, 0, vec![Ty::new_mut_ptr(tcx, param(0)), param(0)], tcx.types.unit)
            }

            sym::typed_swap_nonoverlapping => {
                (1, 0, vec![Ty::new_mut_ptr(tcx, param(0)); 2], tcx.types.unit)
            }

            sym::discriminant_value => {
                let assoc_items = tcx.associated_item_def_ids(
                    tcx.require_lang_item(hir::LangItem::DiscriminantKind, None),
                );
                let discriminant_def_id = assoc_items[0];

                let br =
                    ty::BoundRegion { var: ty::BoundVar::ZERO, kind: ty::BoundRegionKind::Anon };
                (
                    1,
                    0,
                    vec![Ty::new_imm_ref(
                        tcx,
                        ty::Region::new_bound(tcx, ty::INNERMOST, br),
                        param(0),
                    )],
                    Ty::new_projection_from_args(
                        tcx,
                        discriminant_def_id,
                        tcx.mk_args(&[param(0).into()]),
                    ),
                )
            }

            sym::catch_unwind => {
                let mut_u8 = Ty::new_mut_ptr(tcx, tcx.types.u8);
                let try_fn_ty = ty::Binder::dummy(tcx.mk_fn_sig(
                    [mut_u8],
                    tcx.types.unit,
                    false,
                    hir::Safety::Safe,
                    ExternAbi::Rust,
                ));
                let catch_fn_ty = ty::Binder::dummy(tcx.mk_fn_sig(
                    [mut_u8, mut_u8],
                    tcx.types.unit,
                    false,
                    hir::Safety::Safe,
                    ExternAbi::Rust,
                ));
                (
                    0,
                    0,
                    vec![Ty::new_fn_ptr(tcx, try_fn_ty), mut_u8, Ty::new_fn_ptr(tcx, catch_fn_ty)],
                    tcx.types.i32,
                )
            }

            sym::va_start | sym::va_end => {
                (0, 0, vec![mk_va_list_ty(hir::Mutability::Mut).0], tcx.types.unit)
            }

            sym::va_copy => {
                let (va_list_ref_ty, va_list_ty) = mk_va_list_ty(hir::Mutability::Not);
                let va_list_ptr_ty = Ty::new_mut_ptr(tcx, va_list_ty);
                (0, 0, vec![va_list_ptr_ty, va_list_ref_ty], tcx.types.unit)
            }

            sym::va_arg => (1, 0, vec![mk_va_list_ty(hir::Mutability::Mut).0], param(0)),

            sym::nontemporal_store => {
                (1, 0, vec![Ty::new_mut_ptr(tcx, param(0)), param(0)], tcx.types.unit)
            }

            sym::raw_eq => {
                let br =
                    ty::BoundRegion { var: ty::BoundVar::ZERO, kind: ty::BoundRegionKind::Anon };
                let param_ty_lhs =
                    Ty::new_imm_ref(tcx, ty::Region::new_bound(tcx, ty::INNERMOST, br), param(0));
                let br = ty::BoundRegion {
                    var: ty::BoundVar::from_u32(1),
                    kind: ty::BoundRegionKind::Anon,
                };
                let param_ty_rhs =
                    Ty::new_imm_ref(tcx, ty::Region::new_bound(tcx, ty::INNERMOST, br), param(0));
                (1, 0, vec![param_ty_lhs, param_ty_rhs], tcx.types.bool)
            }

            sym::black_box => (1, 0, vec![param(0)], param(0)),

            sym::is_val_statically_known => (1, 0, vec![param(0)], tcx.types.bool),

            sym::const_eval_select => (4, 0, vec![param(0), param(1), param(2)], param(3)),

            sym::vtable_size | sym::vtable_align => {
                (0, 0, vec![Ty::new_imm_ptr(tcx, tcx.types.unit)], tcx.types.usize)
            }

            // This type check is not particularly useful, but the `where` bounds
            // on the definition in `core` do the heavy lifting for checking it.
            sym::aggregate_raw_ptr => (3, 0, vec![param(1), param(2)], param(0)),
            sym::ptr_metadata => (2, 0, vec![Ty::new_imm_ptr(tcx, param(0))], param(1)),

            sym::ub_checks => (0, 0, Vec::new(), tcx.types.bool),

            sym::box_new => (1, 0, vec![param(0)], Ty::new_box(tcx, param(0))),

            // contract_checks() -> bool
            sym::contract_checks => (0, 0, Vec::new(), tcx.types.bool),
            // contract_check_requires::<C>(C) -> bool, where C: impl Fn() -> bool
            sym::contract_check_requires => (1, 0, vec![param(0)], tcx.types.unit),

            sym::simd_eq
            | sym::simd_ne
            | sym::simd_lt
            | sym::simd_le
            | sym::simd_gt
            | sym::simd_ge => (2, 0, vec![param(0), param(0)], param(1)),
            sym::simd_add
            | sym::simd_sub
            | sym::simd_mul
            | sym::simd_rem
            | sym::simd_div
            | sym::simd_shl
            | sym::simd_shr
            | sym::simd_and
            | sym::simd_or
            | sym::simd_xor
            | sym::simd_fmin
            | sym::simd_fmax
            | sym::simd_saturating_add
            | sym::simd_saturating_sub => (1, 0, vec![param(0), param(0)], param(0)),
            sym::simd_arith_offset => (2, 0, vec![param(0), param(1)], param(0)),
            sym::simd_neg
            | sym::simd_bswap
            | sym::simd_bitreverse
            | sym::simd_ctlz
            | sym::simd_cttz
            | sym::simd_ctpop
            | sym::simd_fsqrt
            | sym::simd_fsin
            | sym::simd_fcos
            | sym::simd_fexp
            | sym::simd_fexp2
            | sym::simd_flog2
            | sym::simd_flog10
            | sym::simd_flog
            | sym::simd_fabs
            | sym::simd_ceil
            | sym::simd_floor
            | sym::simd_round
            | sym::simd_trunc => (1, 0, vec![param(0)], param(0)),
            sym::simd_fma | sym::simd_relaxed_fma => {
                (1, 0, vec![param(0), param(0), param(0)], param(0))
            }
            sym::simd_gather => (3, 0, vec![param(0), param(1), param(2)], param(0)),
            sym::simd_masked_load => (3, 0, vec![param(0), param(1), param(2)], param(2)),
            sym::simd_masked_store => (3, 0, vec![param(0), param(1), param(2)], tcx.types.unit),
            sym::simd_scatter => (3, 0, vec![param(0), param(1), param(2)], tcx.types.unit),
            sym::simd_insert | sym::simd_insert_dyn => {
                (2, 0, vec![param(0), tcx.types.u32, param(1)], param(0))
            }
            sym::simd_extract | sym::simd_extract_dyn => {
                (2, 0, vec![param(0), tcx.types.u32], param(1))
            }
            sym::simd_cast
            | sym::simd_as
            | sym::simd_cast_ptr
            | sym::simd_expose_provenance
            | sym::simd_with_exposed_provenance => (2, 0, vec![param(0)], param(1)),
            sym::simd_bitmask => (2, 0, vec![param(0)], param(1)),
            sym::simd_select | sym::simd_select_bitmask => {
                (2, 0, vec![param(0), param(1), param(1)], param(1))
            }
            sym::simd_reduce_all | sym::simd_reduce_any => (1, 0, vec![param(0)], tcx.types.bool),
            sym::simd_reduce_add_ordered | sym::simd_reduce_mul_ordered => {
                (2, 0, vec![param(0), param(1)], param(1))
            }
            sym::simd_reduce_add_unordered
            | sym::simd_reduce_mul_unordered
            | sym::simd_reduce_and
            | sym::simd_reduce_or
            | sym::simd_reduce_xor
            | sym::simd_reduce_min
            | sym::simd_reduce_max => (2, 0, vec![param(0)], param(1)),
            sym::simd_shuffle => (3, 0, vec![param(0), param(0), param(1)], param(2)),
            sym::simd_shuffle_const_generic => (2, 1, vec![param(0), param(0)], param(1)),

            other => {
                tcx.dcx().emit_err(UnrecognizedIntrinsicFunction { span, name: other });
                return;
            }
        };
        (n_tps, 0, n_cts, inputs, output, safety)
    };
    let sig = tcx.mk_fn_sig(inputs, output, false, safety, ExternAbi::Rust);
    let sig = ty::Binder::bind_with_vars(sig, bound_vars);
    equate_intrinsic_type(tcx, span, intrinsic_id, n_tps, n_lts, n_cts, sig)
}
