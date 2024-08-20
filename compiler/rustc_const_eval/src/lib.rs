// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![doc(rust_logo)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(decl_macro)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(never_type)]
#![feature(rustdoc_internals)]
#![feature(slice_ptr_get)]
#![feature(strict_provenance)]
#![feature(trait_alias)]
#![feature(try_blocks)]
#![feature(yeet_expr)]
// tidy-alphabetical-end

pub mod check_consts;
pub mod const_eval;
mod errors;
pub mod interpret;
pub mod util;

use std::sync::atomic::AtomicBool;

pub use errors::ReportErrorExt;
use rustc_middle::ty;
use rustc_middle::util::Providers;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

pub fn provide(providers: &mut Providers) {
    const_eval::provide(providers);
    providers.tag_for_variant = const_eval::tag_for_variant_provider;
    providers.eval_to_const_value_raw = const_eval::eval_to_const_value_raw_provider;
    providers.eval_to_allocation_raw = const_eval::eval_to_allocation_raw_provider;
    providers.eval_static_initializer = const_eval::eval_static_initializer_provider;
    providers.hooks.const_caller_location = util::caller_location::const_caller_location_provider;
    providers.eval_to_valtree = |tcx, param_env_and_value| {
        let (param_env, raw) = param_env_and_value.into_parts();
        const_eval::eval_to_valtree(tcx, param_env, raw)
    };
    providers.hooks.try_destructure_mir_constant_for_user_output =
        const_eval::try_destructure_mir_constant_for_user_output;
    providers.valtree_to_const_val = |tcx, (ty, valtree)| {
        const_eval::valtree_to_const_value(tcx, ty::ParamEnv::empty().and(ty), valtree)
    };
    providers.check_validity_requirement = |tcx, (init_kind, param_env_and_ty)| {
        util::check_validity_requirement(tcx, init_kind, param_env_and_ty)
    };
}

use rustc_middle::mir::{
    BasicBlockData, ConstOperand, NullOp, Operand, Rvalue, StatementKind, SwitchTargets,
    TerminatorKind,
};
use rustc_middle::ty::{Instance, TyCtxt};

/// `rustc_driver::main` installs a handler that will set this to `true` if
/// the compiler has been sent a request to shut down, such as by a Ctrl-C.
/// This static lives here because it is only read by the interpreter.
pub static CTRL_C_RECEIVED: AtomicBool = AtomicBool::new(false);

/// If this basic block ends with a [`TerminatorKind::SwitchInt`] for which we can evaluate the
/// dimscriminant in monomorphization, we return the discriminant bits and the
/// [`SwitchTargets`], just so the caller doesn't also have to match on the terminator.
pub fn try_const_mono_switchint<'a, 'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    block: &'a BasicBlockData<'tcx>,
) -> Option<(u128, &'a SwitchTargets)> {
    // There are two places here we need to evaluate a constant.
    let eval_mono_const = |constant: &ConstOperand<'tcx>| {
        let env = ty::ParamEnv::reveal_all();
        let mono_literal = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            env,
            crate::ty::EarlyBinder::bind(constant.const_),
        );
        mono_literal.try_eval_bits(tcx, env)
    };

    let TerminatorKind::SwitchInt { discr, targets } = &block.terminator().kind else {
        return None;
    };

    // If this is a SwitchInt(const _), then we can just evaluate the constant and return.
    let discr = match discr {
        Operand::Constant(constant) => {
            let bits = eval_mono_const(constant)?;
            return Some((bits, targets));
        }
        Operand::Move(place) | Operand::Copy(place) => place,
    };

    // MIR for `if false` actually looks like this:
    // _1 = const _
    // SwitchInt(_1)
    //
    // And MIR for if intrinsics::ub_checks() looks like this:
    // _1 = UbChecks()
    // SwitchInt(_1)
    //
    // So we're going to try to recognize this pattern.
    //
    // If we have a SwitchInt on a non-const place, we find the most recent statement that
    // isn't a storage marker. If that statement is an assignment of a const to our
    // discriminant place, we evaluate and return the const, as if we've const-propagated it
    // into the SwitchInt.

    let last_stmt = block.statements.iter().rev().find(|stmt| {
        !matches!(stmt.kind, StatementKind::StorageDead(_) | StatementKind::StorageLive(_))
    })?;

    let (place, rvalue) = last_stmt.kind.as_assign()?;

    if discr != place {
        return None;
    }

    use rustc_span::DUMMY_SP;

    use crate::const_eval::DummyMachine;
    use crate::interpret::InterpCx;
    match rvalue {
        Rvalue::NullaryOp(NullOp::UbChecks, _) => Some((tcx.sess.ub_checks() as u128, targets)),
        Rvalue::Use(Operand::Constant(constant)) => {
            let bits = eval_mono_const(constant)?;
            Some((bits, targets))
        }
        Rvalue::BinaryOp(binop, box (Operand::Constant(lhs), Operand::Constant(rhs))) => {
            let env = ty::ParamEnv::reveal_all();
            let lhs = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                env,
                crate::ty::EarlyBinder::bind(lhs.const_),
            );
            let ecx = InterpCx::new(tcx, DUMMY_SP, env, DummyMachine);
            let lhs = ecx.eval_mir_constant(&lhs, DUMMY_SP, None).ok()?;
            let lhs = ecx.read_immediate(&lhs).ok()?;

            let rhs = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                env,
                crate::ty::EarlyBinder::bind(rhs.const_),
            );
            let rhs = ecx.eval_mir_constant(&rhs, DUMMY_SP, None).ok()?;
            let rhs = ecx.read_immediate(&rhs).ok()?;

            let res = ecx.binary_op(*binop, &lhs, &rhs).ok()?;
            let res = res.to_scalar_int().unwrap().to_bits_unchecked();
            Some((res, targets))
        }
        _ => None,
    }
}
