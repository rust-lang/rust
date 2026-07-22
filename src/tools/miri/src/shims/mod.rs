#![warn(clippy::arithmetic_side_effects)]

mod alloc;
mod backtrace;
mod files;
mod math;
#[cfg(all(feature = "native-lib", unix))]
pub mod native_lib;
mod unix;
mod windows;

pub mod env;
pub mod extern_static;
pub mod foreign_items;
pub mod global_ctor;
pub mod io_error;
pub mod os_str;
pub mod panic;
pub mod readiness;
pub mod sig;
pub mod time;
pub mod tls;
pub mod unwind;

pub use self::files::{FdId, FdTable, FileDescription, FileDescriptionRef, WeakFileDescriptionRef};
#[cfg(all(feature = "native-lib", unix))]
pub use self::native_lib::trace::{init_sv, register_retcode_sv};
pub use self::unix::DirTable;

/// What needs to be done after emulating an item (a shim or an intrinsic) is done.
pub enum EmulateItemResult {
    /// The caller is expected to jump to the return block.
    NeedsReturn,
    /// The caller is expected to jump to the unwind block.
    NeedsUnwind,
    /// Jumping to the next block has already been taken care of.
    AlreadyJumped,
    /// The item is not supported.
    NotSupported,
}

impl EmulateItemResult {
    pub fn jump_to_next_block<'tcx, T: Default>(
        self,
        ecx: &mut crate::MiriInterpCx<'tcx>,
        dest: &crate::MPlaceTy<'tcx>,
        ret: Option<rustc_middle::mir::BasicBlock>,
        unwind: Option<rustc_middle::mir::UnwindAction>,
        not_supported: impl FnOnce(&mut crate::MiriInterpCx<'tcx>) -> crate::InterpResult<'tcx, T>,
    ) -> crate::InterpResult<'tcx, T> {
        use crate::*;

        match self {
            EmulateItemResult::NeedsReturn => {
                trace!("{:?}", ecx.dump_place(&dest.clone().into()));
                ecx.return_to_block(ret)?;
                interp_ok(T::default())
            }
            EmulateItemResult::NeedsUnwind => {
                // Jump to the unwind block to begin unwinding.
                ecx.unwind_to_block(unwind.unwrap())?;
                interp_ok(T::default())
            }
            EmulateItemResult::AlreadyJumped => interp_ok(T::default()),
            EmulateItemResult::NotSupported => not_supported(ecx),
        }
    }
}
