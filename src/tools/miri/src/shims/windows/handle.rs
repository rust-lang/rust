use rustc_target::abi::HasDataLayout;
use std::mem::variant_count;

use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PseudoHandle {
    CurrentThread,
}

/// Miri representation of a Windows `HANDLE`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Handle {
    Null,
    Pseudo(PseudoHandle),
    Thread(ThreadId),
}

impl PseudoHandle {
    const CURRENT_THREAD_VALUE: u32 = 0;

    fn value(self) -> u32 {
        match self {
            Self::CurrentThread => Self::CURRENT_THREAD_VALUE,
        }
    }

    fn from_value(value: u32) -> Option<Self> {
        match value {
            Self::CURRENT_THREAD_VALUE => Some(Self::CurrentThread),
            _ => None,
        }
    }
}

impl Handle {
    const NULL_DISCRIMINANT: u32 = 0;
    const PSEUDO_DISCRIMINANT: u32 = 1;
    const THREAD_DISCRIMINANT: u32 = 2;

    fn discriminant(self) -> u32 {
        match self {
            Self::Null => Self::NULL_DISCRIMINANT,
            Self::Pseudo(_) => Self::PSEUDO_DISCRIMINANT,
            Self::Thread(_) => Self::THREAD_DISCRIMINANT,
        }
    }

    fn data(self) -> u32 {
        match self {
            Self::Null => 0,
            Self::Pseudo(pseudo_handle) => pseudo_handle.value(),
            Self::Thread(thread) => thread.to_u32(),
        }
    }

    fn packed_disc_size() -> u32 {
        // ceil(log2(x)) is how many bits it takes to store x numbers
        let variant_count = variant_count::<Self>();

        // however, std's ilog2 is floor(log2(x))
        let floor_log2 = variant_count.ilog2();

        // we need to add one for non powers of two to compensate for the difference
        #[allow(clippy::integer_arithmetic)] // cannot overflow
        if variant_count.is_power_of_two() { floor_log2 } else { floor_log2 + 1 }
    }

    /// Converts a handle into its machine representation.
    ///
    /// The upper [`Self::packed_disc_size()`] bits are used to store a discriminant corresponding to the handle variant.
    /// The remaining bits are used for the variant's field.
    ///
    /// None of this layout is guaranteed to applications by Windows or Miri.
    fn to_packed(self) -> u32 {
        let disc_size = Self::packed_disc_size();
        let data_size = u32::BITS.checked_sub(disc_size).unwrap();

        let discriminant = self.discriminant();
        let data = self.data();

        // make sure the discriminant fits into `disc_size` bits
        assert!(discriminant < 2u32.pow(disc_size));

        // make sure the data fits into `data_size` bits
        assert!(data < 2u32.pow(data_size));

        // packs the data into the lower `data_size` bits
        // and packs the discriminant right above the data
        #[allow(clippy::integer_arithmetic)] // cannot overflow
        return discriminant << data_size | data;
    }

    fn new(discriminant: u32, data: u32) -> Option<Self> {
        match discriminant {
            Self::NULL_DISCRIMINANT if data == 0 => Some(Self::Null),
            Self::PSEUDO_DISCRIMINANT => Some(Self::Pseudo(PseudoHandle::from_value(data)?)),
            Self::THREAD_DISCRIMINANT => Some(Self::Thread(data.into())),
            _ => None,
        }
    }

    /// see docs for `to_packed`
    fn from_packed(handle: u32) -> Option<Self> {
        let disc_size = Self::packed_disc_size();
        let data_size = u32::BITS.checked_sub(disc_size).unwrap();

        // the lower `data_size` bits of this mask are 1
        #[allow(clippy::integer_arithmetic)] // cannot overflow
        let data_mask = 2u32.pow(data_size) - 1;

        // the discriminant is stored right above the lower `data_size` bits
        #[allow(clippy::integer_arithmetic)] // cannot overflow
        let discriminant = handle >> data_size;

        // the data is stored in the lower `data_size` bits
        let data = handle & data_mask;

        Self::new(discriminant, data)
    }

    pub fn to_scalar(self, cx: &impl HasDataLayout) -> Scalar<Provenance> {
        // 64-bit handles are sign extended 32-bit handles
        // see https://docs.microsoft.com/en-us/windows/win32/winprog64/interprocess-communication
        #[allow(clippy::cast_possible_wrap)] // we want it to wrap
        let signed_handle = self.to_packed() as i32;
        Scalar::from_target_isize(signed_handle.into(), cx)
    }

    pub fn from_scalar<'tcx>(
        handle: Scalar<Provenance>,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Option<Self>> {
        let sign_extended_handle = handle.to_target_isize(cx)?;

        #[allow(clippy::cast_sign_loss)] // we want to lose the sign
        let handle = if let Ok(signed_handle) = i32::try_from(sign_extended_handle) {
            signed_handle as u32
        } else {
            // if a handle doesn't fit in an i32, it isn't valid.
            return Ok(None);
        };

        Ok(Self::from_packed(handle))
    }
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}

#[allow(non_snake_case)]
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn invalid_handle(&mut self, function_name: &str) -> InterpResult<'tcx, !> {
        throw_machine_stop!(TerminationInfo::Abort(format!(
            "invalid handle passed to `{function_name}`"
        )))
    }

    fn CloseHandle(&mut self, handle_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let handle = this.read_scalar(handle_op)?;

        match Handle::from_scalar(handle, this)? {
            Some(Handle::Thread(thread)) =>
                this.detach_thread(thread, /*allow_terminated_joined*/ true)?,
            _ => this.invalid_handle("CloseHandle")?,
        }

        Ok(())
    }
}
