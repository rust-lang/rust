#![warn(clippy::arithmetic_side_effects)]

mod aarch64;
mod atomic;
mod loongarch;
mod math;
mod simd;
mod x86;

pub use self::atomic::AtomicRmwOp;

#[rustfmt::skip] // prevent `use` reordering
use rand::RngExt;
use rustc_abi::{Endian, Size};
use rustc_middle::{mir, ty};
use rustc_span::{Symbol, sym};
use rustc_target::spec::Arch;

use self::atomic::EvalContextExt as _;
use self::math::EvalContextExt as _;
use self::simd::EvalContextExt as _;
use crate::*;

/// Check that the number of args is what we expect.
fn check_intrinsic_arg_count<'a, 'tcx, const N: usize>(
    args: &'a [OpTy<'tcx>],
) -> InterpResult<'tcx, &'a [OpTy<'tcx>; N]>
where
    &'a [OpTy<'tcx>; N]: TryFrom<&'a [OpTy<'tcx>]>,
{
    if let Ok(ops) = args.try_into() {
        return interp_ok(ops);
    }
    throw_ub_format!(
        "incorrect number of arguments for intrinsic: got {}, expected {}",
        args.len(),
        N
    )
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn call_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx, Option<ty::Instance<'tcx>>> {
        let this = self.eval_context_mut();

        // See if the core engine can handle this intrinsic.
        if this.eval_intrinsic(instance, args, dest, ret)? {
            return interp_ok(None);
        }
        let intrinsic_name = this.tcx.item_name(instance.def_id());
        let intrinsic_name = intrinsic_name.as_str();

        // FIXME: avoid allocating memory
        let dest = this.force_allocation(dest)?;

        let res =
            this.emulate_intrinsic_by_name(intrinsic_name, instance.args, args, &dest, ret)?;
        res.jump_to_next_block(this, &dest, ret, unwind, |this| {
            // We haven't handled the intrinsic, let's see if we can use a fallback body.
            if this.tcx.intrinsic(instance.def_id()).unwrap().must_be_overridden {
                throw_unsup_format!("unimplemented intrinsic: `{intrinsic_name}`")
            }
            let intrinsic_fallback_is_spec = Symbol::intern("intrinsic_fallback_is_spec");
            if this
                .tcx
                .get_attrs_by_path(instance.def_id(), &[sym::miri, intrinsic_fallback_is_spec])
                .next()
                .is_none()
            {
                throw_unsup_format!(
                    "Miri can only use intrinsic fallback bodies that exactly reflect the specification: they fully check for UB and are as non-deterministic as possible. After verifying that `{intrinsic_name}` does so, add the `#[miri::intrinsic_fallback_is_spec]` attribute to it; also ping @rust-lang/miri when you do that"
                );
            }
            interp_ok(Some(ty::Instance {
                def: ty::InstanceKind::Item(instance.def_id()),
                args: instance.args,
            }))
        })
    }

    /// Emulates a Miri-supported intrinsic (not supported by the core engine).
    /// Returns `Ok(true)` if the intrinsic was handled.
    fn emulate_intrinsic_by_name(
        &mut self,
        intrinsic_name: &str,
        generic_args: ty::GenericArgsRef<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        if let Some(name) = intrinsic_name.strip_prefix("atomic_") {
            return this.emulate_atomic_intrinsic(name, generic_args, args, dest);
        }
        if let Some(name) = intrinsic_name.strip_prefix("simd_") {
            return this.emulate_simd_intrinsic(name, args, dest);
        }

        match intrinsic_name {
            // Basic control flow
            "abort" => {
                throw_machine_stop!(TerminationInfo::Abort(
                    "the program aborted execution".to_owned()
                ));
            }
            "catch_unwind" => {
                let [try_fn, data, catch_fn] = check_intrinsic_arg_count(args)?;
                this.handle_catch_unwind(try_fn, data, catch_fn, dest, ret)?;
                // This pushed a stack frame, don't jump to `ret`.
                return interp_ok(EmulateItemResult::AlreadyJumped);
            }

            // Memory model / provenance manipulation
            "ptr_mask" => {
                let [ptr, mask] = check_intrinsic_arg_count(args)?;

                let ptr = this.read_pointer(ptr)?;
                let mask = this.read_target_usize(mask)?;

                let masked_addr = Size::from_bytes(ptr.addr().bytes() & mask);

                this.write_pointer(Pointer::new(ptr.provenance, masked_addr), dest)?;
            }

            // We want to return either `true` or `false` at random, or else something like
            // ```
            // if !is_val_statically_known(0) { unreachable_unchecked(); }
            // ```
            // Would not be considered UB, or the other way around (`is_val_statically_known(0)`).
            "is_val_statically_known" => {
                let [_arg] = check_intrinsic_arg_count(args)?;
                // FIXME: should we check for validity here? It's tricky because we do not have a
                // place. Codegen does not seem to set any attributes like `noundef` for intrinsic
                // calls, so we don't *have* to do anything.
                let branch: bool = this.machine.rng.get_mut().random();
                this.write_scalar(Scalar::from_bool(branch), dest)?;
            }

            // Other
            "breakpoint" => {
                let [] = check_intrinsic_arg_count(args)?;
                // normally this would raise a SIGTRAP, which aborts if no debugger is connected
                throw_machine_stop!(TerminationInfo::Abort(format!("trace/breakpoint trap")))
            }

            "assert_inhabited" | "assert_zero_valid" | "assert_mem_uninitialized_valid" => {
                // Make these a NOP, so we get the better Miri-native error messages.
            }

            _ => return this.emulate_math_intrinsic(intrinsic_name, generic_args, args, dest),
        }

        interp_ok(EmulateItemResult::NeedsReturn)
    }

    fn call_llvm_intrinsic(
        &mut self,
        instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let link_name = this.tcx.codegen_fn_attrs(instance.def_id()).symbol_name.unwrap();

        // FIXME: avoid allocating memory
        let dest = this.force_allocation(dest)?;

        let handled = match link_name.as_str() {
            // LLVM intrinsics
            "llvm.prefetch.p0" => {
                let [p, rw, loc, ty] = this.check_shim_sig_unadjusted(link_name, args)?;

                let _ = this.read_pointer(p)?;
                let rw = this.read_scalar(rw)?.to_i32()?;
                let loc = this.read_scalar(loc)?.to_i32()?;
                let ty = this.read_scalar(ty)?.to_i32()?;

                if ty == 1 {
                    // Data cache prefetch.
                    // Notably, we do not have to check the pointer, this operation is never UB!

                    if !matches!(rw, 0 | 1) {
                        throw_unsup_format!("invalid `rw` value passed to `llvm.prefetch`: {}", rw);
                    }
                    if !matches!(loc, 0..=3) {
                        throw_unsup_format!(
                            "invalid `loc` value passed to `llvm.prefetch`: {}",
                            loc
                        );
                    }
                } else {
                    throw_unsup_format!("unsupported `llvm.prefetch` type argument: {}", ty);
                }

                true
            }
            // Used to implement the x86 `_mm{,256,512}_popcnt_epi{8,16,32,64}` and wasm
            // `{i,u}8x16_popcnt` functions.
            name if name.starts_with("llvm.ctpop.v")
                && this.tcx.sess.target.endian == Endian::Little =>
            {
                let [op] = this.check_shim_sig_unadjusted(link_name, args)?;

                let (op, op_len) = this.project_to_simd(op)?;
                let (dest, dest_len) = this.project_to_simd(&dest)?;

                assert_eq!(dest_len, op_len);

                for i in 0..dest_len {
                    let op = this.read_immediate(&this.project_index(&op, i)?)?;
                    // Use `to_uint` to get a zero-extended `u128`. Those
                    // extra zeros will not affect `count_ones`.
                    let res = op.to_scalar().to_uint(op.layout.size)?.count_ones();

                    this.write_scalar(
                        Scalar::from_uint(res, op.layout.size),
                        &this.project_index(&dest, i)?,
                    )?;
                }

                true
            }

            // Target-specific shims
            name if name.starts_with("llvm.x86.")
                && matches!(this.tcx.sess.target.arch, Arch::X86 | Arch::X86_64)
                && this.tcx.sess.target.endian == Endian::Little =>
                x86::EvalContextExt::emulate_x86_intrinsic(this, link_name, args, &dest)?,
            name if name.starts_with("llvm.aarch64.")
                && this.tcx.sess.target.arch == Arch::AArch64
                && this.tcx.sess.target.endian == Endian::Little =>
                aarch64::EvalContextExt::emulate_aarch64_intrinsic(this, link_name, args, &dest)?,
            name if name.starts_with("llvm.loongarch.")
                && matches!(this.tcx.sess.target.arch, Arch::LoongArch32 | Arch::LoongArch64)
                && this.tcx.sess.target.endian == Endian::Little =>
                loongarch::EvalContextExt::emulate_loongarch_intrinsic(
                    this, link_name, args, &dest,
                )?,
            _ => false,
        };

        // The rest either implements the logic, or falls back to `lookup_exported_symbol`.
        if handled {
            trace!("{:?}", this.dump_place(&dest.clone().into()));
            this.return_to_block(ret)
        } else {
            throw_machine_stop!(TerminationInfo::UnsupportedForeignItem(format!(
                "can't call LLVM intrinsic `{link_name}` on architecture `{arch}`",
                arch = this.tcx.sess.target.arch,
            )));
        }
    }
}
