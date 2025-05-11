use std::mem::variant_count;

use rustc_abi::HasDataLayout;

use crate::concurrency::thread::ThreadNotFound;
use crate::shims::files::FdNum;
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
    File(FdNum),
    Invalid,
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

/// Errors that can occur when constructing a [`Handle`] from a Scalar.
pub enum HandleError {
    /// There is no thread with the given ID.
    ThreadNotFound(ThreadNotFound),
    /// Can't convert scalar to handle because it is structurally invalid.
    InvalidHandle,
}

impl Handle {
    const NULL_DISCRIMINANT: u32 = 0;
    const PSEUDO_DISCRIMINANT: u32 = 1;
    const THREAD_DISCRIMINANT: u32 = 2;
    const FILE_DISCRIMINANT: u32 = 3;
    // Chosen to ensure Handle::Invalid encodes to -1. Update this value if there are ever more than
    // 8 discriminants.
    const INVALID_DISCRIMINANT: u32 = 7;

    fn discriminant(self) -> u32 {
        match self {
            Self::Null => Self::NULL_DISCRIMINANT,
            Self::Pseudo(_) => Self::PSEUDO_DISCRIMINANT,
            Self::Thread(_) => Self::THREAD_DISCRIMINANT,
            Self::File(_) => Self::FILE_DISCRIMINANT,
            Self::Invalid => Self::INVALID_DISCRIMINANT,
        }
    }

    fn data(self) -> u32 {
        match self {
            Self::Null => 0,
            Self::Pseudo(pseudo_handle) => pseudo_handle.value(),
            Self::Thread(thread) => thread.to_u32(),
            #[expect(clippy::cast_sign_loss)]
            Self::File(fd) => fd as u32,
            // INVALID_HANDLE_VALUE is -1. This fact is explicitly declared or implied in several
            // pages of Windows documentation.
            // 1: https://learn.microsoft.com/en-us/dotnet/api/microsoft.win32.safehandles.safefilehandle?view=net-9.0
            // 2: https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/get-osfhandle?view=msvc-170
            Self::Invalid => 0x1FFFFFFF,
        }
    }

    fn packed_disc_size() -> u32 {
        // ceil(log2(x)) is how many bits it takes to store x numbers.
        // We ensure that INVALID_HANDLE_VALUE (0xFFFFFFFF) decodes to Handle::Invalid.
        // see https://devblogs.microsoft.com/oldnewthing/20230914-00/?p=108766 for more detail on
        // INVALID_HANDLE_VALUE.
        let variant_count = variant_count::<Self>();

        // However, std's ilog2 is floor(log2(x)).
        let floor_log2 = variant_count.ilog2();

        // We need to add one for non powers of two to compensate for the difference.
        #[expect(clippy::arithmetic_side_effects)] // cannot overflow
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
        let data_size = u32::BITS.strict_sub(disc_size);

        let discriminant = self.discriminant();
        let data = self.data();

        // make sure the discriminant fits into `disc_size` bits
        assert!(discriminant < 2u32.pow(disc_size));

        // make sure the data fits into `data_size` bits
        assert!(data < 2u32.pow(data_size));

        // packs the data into the lower `data_size` bits
        // and packs the discriminant right above the data
        (discriminant << data_size) | data
    }

    fn new(discriminant: u32, data: u32) -> Option<Self> {
        match discriminant {
            Self::NULL_DISCRIMINANT if data == 0 => Some(Self::Null),
            Self::PSEUDO_DISCRIMINANT => Some(Self::Pseudo(PseudoHandle::from_value(data)?)),
            Self::THREAD_DISCRIMINANT => Some(Self::Thread(ThreadId::new_unchecked(data))),
            #[expect(clippy::cast_possible_wrap)]
            Self::FILE_DISCRIMINANT => {
                // This cast preserves all bits.
                assert_eq!(size_of_val(&data), size_of::<FdNum>());
                Some(Self::File(data as FdNum))
            }
            Self::INVALID_DISCRIMINANT => Some(Self::Invalid),
            _ => None,
        }
    }

    /// see docs for `to_packed`
    fn from_packed(handle: u32) -> Option<Self> {
        let disc_size = Self::packed_disc_size();
        let data_size = u32::BITS.strict_sub(disc_size);

        // the lower `data_size` bits of this mask are 1
        #[expect(clippy::arithmetic_side_effects)] // cannot overflow
        let data_mask = 2u32.pow(data_size) - 1;

        // the discriminant is stored right above the lower `data_size` bits
        let discriminant = handle >> data_size;

        // the data is stored in the lower `data_size` bits
        let data = handle & data_mask;

        Self::new(discriminant, data)
    }

    pub fn to_scalar(self, cx: &impl HasDataLayout) -> Scalar {
        // 64-bit handles are sign extended 32-bit handles
        // see https://docs.microsoft.com/en-us/windows/win32/winprog64/interprocess-communication
        #[expect(clippy::cast_possible_wrap)] // we want it to wrap
        let signed_handle = self.to_packed() as i32;
        Scalar::from_target_isize(signed_handle.into(), cx)
    }

    /// Convert a scalar into a structured `Handle`.
    /// Structurally invalid handles return [`HandleError::InvalidHandle`].
    /// If the handle is structurally valid but semantically invalid, e.g. a for non-existent thread
    /// ID, returns [`HandleError::ThreadNotFound`].
    fn try_from_scalar<'tcx>(
        handle: Scalar,
        cx: &MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Result<Self, HandleError>> {
        let sign_extended_handle = handle.to_target_isize(cx)?;

        #[expect(clippy::cast_sign_loss)] // we want to lose the sign
        let handle = if let Ok(signed_handle) = i32::try_from(sign_extended_handle) {
            signed_handle as u32
        } else {
            // if a handle doesn't fit in an i32, it isn't valid.
            return interp_ok(Err(HandleError::InvalidHandle));
        };

        match Self::from_packed(handle) {
            Some(Self::Thread(thread)) => {
                // validate the thread id
                match cx.machine.threads.thread_id_try_from(thread.to_u32()) {
                    Ok(id) => interp_ok(Ok(Self::Thread(id))),
                    Err(e) => interp_ok(Err(HandleError::ThreadNotFound(e))),
                }
            }
            Some(handle) => interp_ok(Ok(handle)),
            None => interp_ok(Err(HandleError::InvalidHandle)),
        }
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}

#[allow(non_snake_case)]
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Convert a scalar into a structured `Handle`.
    /// If the handle is invalid, or references a non-existent item, execution is aborted.
    #[track_caller]
    fn read_handle(&self, handle: &OpTy<'tcx>, function_name: &str) -> InterpResult<'tcx, Handle> {
        let this = self.eval_context_ref();
        let handle = this.read_scalar(handle)?;
        match Handle::try_from_scalar(handle, this)? {
            Ok(handle) => interp_ok(handle),
            Err(HandleError::InvalidHandle) =>
                throw_machine_stop!(TerminationInfo::Abort(format!(
                    "invalid handle {} passed to {function_name}",
                    handle.to_target_isize(this)?,
                ))),
            Err(HandleError::ThreadNotFound(_)) =>
                throw_machine_stop!(TerminationInfo::Abort(format!(
                    "invalid thread ID {} passed to {function_name}",
                    handle.to_target_isize(this)?,
                ))),
        }
    }

    fn invalid_handle(&mut self, function_name: &str) -> InterpResult<'tcx, !> {
        throw_machine_stop!(TerminationInfo::Abort(format!(
            "invalid handle passed to `{function_name}`"
        )))
    }

    fn CloseHandle(&mut self, handle_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let handle = this.read_handle(handle_op, "CloseHandle")?;
        let ret = match handle {
            Handle::Thread(thread) => {
                this.detach_thread(thread, /*allow_terminated_joined*/ true)?;
                this.eval_windows("c", "TRUE")
            }
            Handle::File(fd_num) =>
                if let Some(fd) = this.machine.fds.remove(fd_num) {
                    let err = fd.close_ref(this.machine.communicate(), this)?;
                    if let Err(e) = err {
                        this.set_last_error(e)?;
                        this.eval_windows("c", "FALSE")
                    } else {
                        this.eval_windows("c", "TRUE")
                    }
                } else {
                    this.invalid_handle("CloseHandle")?
                },
            _ => this.invalid_handle("CloseHandle")?,
        };

        interp_ok(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_encoding() {
        // Ensure the invalid handle encodes to `u32::MAX`/`INVALID_HANDLE_VALUE`.
        assert_eq!(Handle::Invalid.to_packed(), u32::MAX)
    }
}
