#![unstable(feature = "motor_ext", issue = "147456")]

use crate::sealed::Sealed;
use crate::sys_common::AsInner;

pub trait ChildExt: Sealed {
    /// Extracts the main thread raw handle, without taking ownership
    fn sys_handle(&self) -> u64;
}

impl ChildExt for crate::process::Child {
    fn sys_handle(&self) -> u64 {
        self.as_inner().handle()
    }
}
