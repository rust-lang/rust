#![unstable(feature = "motor_ext", issue = "147456")]

use crate::sys::AsInner;

pub impl(self) trait ChildExt {
    /// Extracts the main thread raw handle, without taking ownership
    fn sys_handle(&self) -> u64;
}

impl ChildExt for crate::process::Child {
    fn sys_handle(&self) -> u64 {
        self.as_inner().handle()
    }
}
