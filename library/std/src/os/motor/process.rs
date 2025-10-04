use crate::sealed::Sealed;
use crate::sys_common::AsInner;

#[unstable(feature = "motor_ext", issue = "none")]
pub trait ChildExt: Sealed {
    /// Extracts the main thread raw handle, without taking ownership
    #[unstable(feature = "motor_ext", issue = "none")]
    fn sys_handle(&self) -> u64;
}

#[unstable(feature = "motor_ext", issue = "none")]
impl ChildExt for crate::process::Child {
    fn sys_handle(&self) -> u64 {
        self.as_inner().handle()
    }
}
