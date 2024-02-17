use std::iter;

use rand::{seq::IteratorRandom, Rng};
use rustc_apfloat::{Float, FloatConvert};
use rustc_middle::mir;
use rustc_target::abi::Size;

use crate::*;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn binary_ptr_op(
        &self,
        bin_op: mir::BinOp,
        left: &ImmTy<'tcx, Provenance>,
        right: &ImmTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, (ImmTy<'tcx, Provenance>, bool)> {
        use rustc_middle::mir::BinOp::*;

        let this = self.eval_context_ref();
        trace!("ptr_op: {:?} {:?} {:?}", *left, bin_op, *right);

        Ok(match bin_op {
            Eq | Ne | Lt | Le | Gt | Ge => {
                assert_eq!(left.layout.abi, right.layout.abi); // types can differ, e.g. fn ptrs with different `for`
                let size = this.pointer_size();
                // Just compare the bits. ScalarPairs are compared lexicographically.
                // We thus always compare pairs and simply fill scalars up with 0.
                let left = match **left {
                    Immediate::Scalar(l) => (l.to_bits(size)?, 0),
                    Immediate::ScalarPair(l1, l2) => (l1.to_bits(size)?, l2.to_bits(size)?),
                    Immediate::Uninit => panic!("we should never see uninit data here"),
                };
                let right = match **right {
                    Immediate::Scalar(r) => (r.to_bits(size)?, 0),
                    Immediate::ScalarPair(r1, r2) => (r1.to_bits(size)?, r2.to_bits(size)?),
                    Immediate::Uninit => panic!("we should never see uninit data here"),
                };
                let res = match bin_op {
                    Eq => left == right,
                    Ne => left != right,
                    Lt => left < right,
                    Le => left <= right,
                    Gt => left > right,
                    Ge => left >= right,
                    _ => bug!(),
                };
                (ImmTy::from_bool(res, *this.tcx), false)
            }

            // Some more operations are possible with atomics.
            // The return value always has the provenance of the *left* operand.
            Add | Sub | BitOr | BitAnd | BitXor => {
                assert!(left.layout.ty.is_unsafe_ptr());
                assert!(right.layout.ty.is_unsafe_ptr());
                let ptr = left.to_scalar().to_pointer(this)?;
                // We do the actual operation with usize-typed scalars.
                let left = ImmTy::from_uint(ptr.addr().bytes(), this.machine.layouts.usize);
                let right = ImmTy::from_uint(
                    right.to_scalar().to_target_usize(this)?,
                    this.machine.layouts.usize,
                );
                let (result, overflowing) = this.overflowing_binary_op(bin_op, &left, &right)?;
                // Construct a new pointer with the provenance of `ptr` (the LHS).
                let result_ptr = Pointer::new(
                    ptr.provenance,
                    Size::from_bytes(result.to_scalar().to_target_usize(this)?),
                );
                (
                    ImmTy::from_scalar(Scalar::from_maybe_pointer(result_ptr, this), left.layout),
                    overflowing,
                )
            }

            _ => span_bug!(this.cur_span(), "Invalid operator on pointers: {:?}", bin_op),
        })
    }

    fn generate_nan<F1: Float + FloatConvert<F2>, F2: Float>(&self, inputs: &[F1]) -> F2 {
        /// Make the given NaN a signaling NaN.
        /// Returns `None` if this would not result in a NaN.
        fn make_signaling<F: Float>(f: F) -> Option<F> {
            // The quiet/signaling bit is the leftmost bit in the mantissa.
            // That's position `PRECISION-1`, since `PRECISION` includes the fixed leading 1 bit,
            // and then we subtract 1 more since this is 0-indexed.
            let quiet_bit_mask = 1 << (F::PRECISION - 2);
            // Unset the bit. Double-check that this wasn't the last bit set in the payload.
            // (which would turn the NaN into an infinity).
            let f = F::from_bits(f.to_bits() & !quiet_bit_mask);
            if f.is_nan() { Some(f) } else { None }
        }

        let this = self.eval_context_ref();
        let mut rand = this.machine.rng.borrow_mut();
        // Assemble an iterator of possible NaNs: preferred, quieting propagation, unchanged propagation.
        // On some targets there are more possibilities; for now we just generate those options that
        // are possible everywhere.
        let preferred_nan = F2::qnan(Some(0));
        let nans = iter::once(preferred_nan)
            .chain(inputs.iter().filter(|f| f.is_nan()).map(|&f| {
                // Regular apfloat cast is quieting.
                f.convert(&mut false).value
            }))
            .chain(inputs.iter().filter(|f| f.is_signaling()).filter_map(|&f| {
                let f: F2 = f.convert(&mut false).value;
                // We have to de-quiet this again for unchanged propagation.
                make_signaling(f)
            }));
        // Pick one of the NaNs.
        let nan = nans.choose(&mut *rand).unwrap();
        // Non-deterministically flip the sign.
        if rand.gen() {
            // This will properly flip even for NaN.
            -nan
        } else {
            nan
        }
    }

    fn adjust_nan<F1: Float + FloatConvert<F2>, F2: Float>(&self, f: F2, inputs: &[F1]) -> F2 {
        if f.is_nan() { self.generate_nan(inputs) } else { f }
    }
}
