use rustc_target::abi::HasDataLayout;
use std::mem::variant_count;

use crate::*;

/// A Windows `HANDLE` that represents a resource instead of being null or a pseudohandle.
///
/// This is a seperate type from [`Handle`] to simplify the packing and unpacking code.
#[derive(Clone, Copy)]
enum RealHandle {
    Thread(ThreadId),
}

impl RealHandle {
    const USABLE_BITS: u32 = 31;

    const THREAD_DISCRIMINANT: u32 = 1;

    fn discriminant(self) -> u32 {
        match self {
            // can't use zero here because all zero handle is invalid
            Self::Thread(_) => Self::THREAD_DISCRIMINANT,
        }
    }

    fn data(self) -> u32 {
        match self {
            Self::Thread(thread) => thread.to_u32(),
        }
    }

    fn packed_disc_size() -> u32 {
        // log2(x) + 1 is how many bits it takes to store x
        // because the discriminants start at 1, the variant count is equal to the highest discriminant
        variant_count::<Self>().ilog2() + 1
    }

    /// This function packs the discriminant and data values into a 31-bit space.
    /// None of this layout is guaranteed to applications by Windows or Miri.
    /// The sign bit is not used to avoid overlapping any pseudo-handles.
    fn to_packed(self) -> i32 {
        let disc_size = Self::packed_disc_size();
        let data_size = Self::USABLE_BITS - disc_size;

        let discriminant = self.discriminant();
        let data = self.data();

        // make sure the discriminant fits into `disc_size` bits
        assert!(discriminant < 2u32.pow(disc_size));

        // make sure the data fits into `data_size` bits
        assert!(data < 2u32.pow(data_size));

        // packs the data into the lower `data_size` bits
        // and packs the discriminant right above the data
        (discriminant << data_size | data) as i32
    }

    fn new(discriminant: u32, data: u32) -> Option<Self> {
        match discriminant {
            Self::THREAD_DISCRIMINANT => Some(Self::Thread(data.into())),
            _ => None,
        }
    }

    /// see docs for `to_packed`
    fn from_packed(handle: i32) -> Option<Self> {
        let handle_bits = handle as u32;

        let disc_size = Self::packed_disc_size();
        let data_size = Self::USABLE_BITS - disc_size;

        // the lower `data_size` bits of this mask are 1
        let data_mask = 2u32.pow(data_size) - 1;

        // the discriminant is stored right above the lower `data_size` bits
        let discriminant = handle_bits >> data_size;

        // the data is stored in the lower `data_size` bits
        let data = handle_bits & data_mask;

        Self::new(discriminant, data)
    }
}

/// Miri representation of a Windows `HANDLE`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Handle {
    Null, // = 0

    // pseudo-handles
    // The lowest real windows pseudo-handle is -6, so miri pseduo-handles start at -7 to break code hardcoding these values
    CurrentThread, // = -7

    // real handles
    Thread(ThreadId),
}

impl Handle {
    const CURRENT_THREAD_VALUE: i32 = -7;

    fn to_packed(self) -> i32 {
        match self {
            Self::Null => 0,
            Self::CurrentThread => Self::CURRENT_THREAD_VALUE,
            Self::Thread(thread) => RealHandle::Thread(thread).to_packed(),
        }
    }

    pub fn to_scalar(self, cx: &impl HasDataLayout) -> Scalar<Provenance> {
        // 64-bit handles are sign extended 32-bit handles
        // see https://docs.microsoft.com/en-us/windows/win32/winprog64/interprocess-communication
        let handle = self.to_packed().into();

        Scalar::from_machine_isize(handle, cx)
    }

    fn from_packed(handle: i64) -> Option<Self> {
        let current_thread_val = Self::CURRENT_THREAD_VALUE as i64;

        if handle == 0 {
            Some(Self::Null)
        } else if handle == current_thread_val {
            Some(Self::CurrentThread)
        } else if let Ok(handle) = handle.try_into() {
            match RealHandle::from_packed(handle)? {
                RealHandle::Thread(id) => Some(Self::Thread(id)),
            }
        } else {
            // if a handle doesn't fit in an i32, it isn't valid.
            None
        }
    }

    pub fn from_scalar<'tcx>(
        handle: Scalar<Provenance>,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Option<Self>> {
        let handle = handle.to_machine_isize(cx)?;

        Ok(Self::from_packed(handle))
    }
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}

#[allow(non_snake_case)]
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn CloseHandle(&mut self, handle_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        match Handle::from_scalar(this.read_scalar(handle_op)?.check_init()?, this)? {
            Some(Handle::Thread(thread)) => this.detach_thread(thread)?,
            _ =>
                throw_machine_stop!(TerminationInfo::Abort(
                    "invalid handle passed to `CloseHandle`".into()
                )),
        };

        Ok(())
    }
}
