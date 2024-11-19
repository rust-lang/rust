use std::mem::variant_count;

use rustc_abi::HasDataLayout;

use crate::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PseudoHandle {
    CurrentThread,
    CurrentProcess,
    Stdin,
    Stdout,
    Stderr,
}

/// Miri representation of a Windows `HANDLE`
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Handle {
    Null,
    Pseudo(PseudoHandle),
    Thread(u32),
    File(u32),
    Invalid,
}

impl PseudoHandle {
    const CURRENT_THREAD_VALUE: u32 = 0;
    const STDIN_VALUE: u32 = 1;
    const STDOUT_VALUE: u32 = 2;
    const STDERR_VALUE: u32 = 3;
    const CURRENT_PROCESS_VALUE: u32 = 4;

    fn value(self) -> u32 {
        match self {
            Self::CurrentThread => Self::CURRENT_THREAD_VALUE,
            Self::CurrentProcess => Self::CURRENT_PROCESS_VALUE,
            Self::Stdin => Self::STDIN_VALUE,
            Self::Stdout => Self::STDOUT_VALUE,
            Self::Stderr => Self::STDERR_VALUE,
        }
    }

    fn from_value(value: u32) -> Option<Self> {
        match value {
            Self::CURRENT_THREAD_VALUE => Some(Self::CurrentThread),
            Self::CURRENT_PROCESS_VALUE => Some(Self::CurrentProcess),
            Self::STDIN_VALUE => Some(Self::Stdin),
            Self::STDOUT_VALUE => Some(Self::Stdout),
            Self::STDERR_VALUE => Some(Self::Stderr),
            _ => None,
        }
    }
}

impl Handle {
    const NULL_DISCRIMINANT: u32 = 0;
    const PSEUDO_DISCRIMINANT: u32 = 1;
    const THREAD_DISCRIMINANT: u32 = 2;
    const FILE_DISCRIMINANT: u32 = 3;
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
            Self::Thread(thread) => thread,
            Self::File(fd) => fd,
            Self::Invalid => 0x1FFFFFFF,
        }
    }

    fn packed_disc_size() -> u32 {
        // We ensure that INVALID_HANDLE_VALUE (0xFFFFFFFF) decodes to Handle::Invalid
        // see https://devblogs.microsoft.com/oldnewthing/20230914-00/?p=108766
        #[expect(clippy::arithmetic_side_effects) ] // cannot overflow
        let variant_count = variant_count::<Self>();

        // however, std's ilog2 is floor(log2(x))
        let floor_log2 = variant_count.ilog2();

        // we need to add one for non powers of two to compensate for the difference
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
        assert!(data <= 2u32.pow(data_size));

        // packs the data into the lower `data_size` bits
        // and packs the discriminant right above the data
        discriminant << data_size | data
    }

    fn new(discriminant: u32, data: u32) -> Option<Self> {
        match discriminant {
            Self::NULL_DISCRIMINANT if data == 0 => Some(Self::Null),
            Self::PSEUDO_DISCRIMINANT => Some(Self::Pseudo(PseudoHandle::from_value(data)?)),
            Self::THREAD_DISCRIMINANT => Some(Self::Thread(data)),
            Self::FILE_DISCRIMINANT => Some(Self::File(data)),
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

    pub fn from_scalar<'tcx>(
        handle: Scalar,
        cx: &impl HasDataLayout,
    ) -> InterpResult<'tcx, Option<Self>> {
        let sign_extended_handle = handle.to_target_isize(cx)?;

        #[expect(clippy::cast_sign_loss)] // we want to lose the sign
        let handle = if let Ok(signed_handle) = i32::try_from(sign_extended_handle) {
            signed_handle as u32
        } else {
            // if a handle doesn't fit in an i32, it isn't valid.
            return interp_ok(None);
        };

        interp_ok(Self::from_packed(handle))
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}

#[allow(non_snake_case)]
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn read_handle(&self, handle: &OpTy<'tcx>) -> InterpResult<'tcx, Handle> {
        let this = self.eval_context_ref();
        let handle = this.read_scalar(handle)?;
        match Handle::from_scalar(handle, this)? {
            Some(handle) => interp_ok(handle),
            None => throw_machine_stop!(TerminationInfo::Abort(format!("invalid handle {}", handle.to_target_isize(this)?))),
        }
    }

    fn invalid_handle(&mut self, function_name: &str) -> InterpResult<'tcx, !> {
        throw_machine_stop!(TerminationInfo::Abort(format!(
            "invalid handle passed to `{function_name}`"
        )))
    }

    fn GetStdHandle(&mut self, which: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();
        let which = this.read_scalar(which)?.to_i32()?;

        let stdin = this.eval_windows("c", "STD_INPUT_HANDLE").to_i32()?;
        let stdout = this.eval_windows("c", "STD_OUTPUT_HANDLE").to_i32()?;
        let stderr = this.eval_windows("c", "STD_ERROR_HANDLE").to_i32()?;

        let handle = if which == stdin {
            Handle::Pseudo(PseudoHandle::Stdin)
        } else if which == stdout {
            Handle::Pseudo(PseudoHandle::Stdout)
        } else if which == stderr {
            Handle::Pseudo(PseudoHandle::Stderr)
        } else {
            throw_unsup_format!("Invalid argument to `GetStdHandle`: {which}")
        };
        interp_ok(handle.to_scalar(this))
    }

    fn CloseHandle(&mut self, handle_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let ret = match this.read_handle(handle_op)? {
            Handle::Thread(thread) => {
                if let Ok(thread) = this.thread_id_try_from(thread) {
                    this.detach_thread(thread, /*allow_terminated_joined*/ true)?;
                    this.eval_windows("c", "TRUE")
                } else {
                    this.invalid_handle("CloseHandle")?
                }
            }
            Handle::File(fd) => {
                if let Some(file) = this.machine.fds.get(fd as i32) {
                    let err = file.close(this.machine.communicate(), this)?;
                    if let Err(e) = err {
                        this.set_last_error(e)?;
                        this.eval_windows("c", "FALSE")
                    } else {
                        this.eval_windows("c", "TRUE")
                    }
                } else {
                    this.invalid_handle("CloseHandle")?
                }
            }
            _ => this.invalid_handle("CloseHandle")?,
        };

        interp_ok(ret)
    }
}
