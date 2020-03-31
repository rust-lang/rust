//! Windows-specific extensions to access Windows profileapi.

#![stable(feature = "windows_profileapi", since = "1.43.0")]

use crate::sys::c;
use crate::sys::cvt;

/// Stores the current value of the performance counter to the memory pointed by the argument pointer,
/// Counter value is a high resolution (<1us) time stamp that can be used for time-interval measurements.
#[stable(feature = "windows_profileapi", since = "1.43.0")]
pub fn QueryPerformanceCounter(qpc_value: &mut i64) -> crate::io::Result<i32> {
    cvt(unsafe { c::QueryPerformanceCounter(qpc_value) })
}

/// Stores the frequency of the performance counter to the memory pointed by the argument pointer.
/// The frequency of the performance counter is fixed at system boot and is consistent across all processors.
/// Therefore, the frequency need only be queried upon application initialization, and the result can be cached.
#[stable(feature = "windows_profileapi", since = "1.43.0")]
pub fn QueryPerformanceFrequency(frequency: &mut i64) -> crate::io::Result<i32> {
    cvt(unsafe { c::QueryPerformanceFrequency(frequency) })
}
