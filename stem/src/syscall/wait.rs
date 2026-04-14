use crate::time::Duration;
use abi::errors::Errno;
use abi::wait::{WaitResult, WaitSpec};

use super::arch::raw_syscall6;
use abi::syscall::SYS_WAIT_MANY;

pub fn wait_many(
    specs: &[WaitSpec],
    results: &mut [WaitResult],
    timeout: Option<Duration>,
) -> Result<usize, Errno> {
    let timeout_ns = timeout.map(|d| d.as_nanos()).unwrap_or(u64::MAX);
    let ret = unsafe {
        raw_syscall6(
            SYS_WAIT_MANY,
            specs.as_ptr() as usize,
            specs.len(),
            results.as_mut_ptr() as usize,
            results.len(),
            timeout_ns as usize,
            0,
        )
    };
    abi::errors::errno(ret)
}
