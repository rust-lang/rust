use super::{
    abi,
    error::{fail, fail_aborting, ItronError},
};

use crate::mem::MaybeUninit;

/// Get the ID of the task in Running state. Panics on failure.
#[inline]
pub fn current_task_id() -> abi::ID {
    try_current_task_id().unwrap_or_else(|e| fail(e, &"get_tid"))
}

/// Get the ID of the task in Running state. Aborts on failure.
#[inline]
pub fn current_task_id_aborting() -> abi::ID {
    try_current_task_id().unwrap_or_else(|e| fail_aborting(e, &"get_tid"))
}

/// Get the ID of the task in Running state.
#[inline]
pub fn try_current_task_id() -> Result<abi::ID, ItronError> {
    unsafe {
        let mut out = MaybeUninit::uninit();
        ItronError::err_if_negative(abi::get_tid(out.as_mut_ptr()))?;
        Ok(out.assume_init())
    }
}

/// Get the specified task's priority. Panics on failure.
#[inline]
pub fn task_priority(task: abi::ID) -> abi::PRI {
    try_task_priority(task).unwrap_or_else(|e| fail(e, &"get_pri"))
}

/// Get the specified task's priority.
#[inline]
pub fn try_task_priority(task: abi::ID) -> Result<abi::PRI, ItronError> {
    unsafe {
        let mut out = MaybeUninit::uninit();
        ItronError::err_if_negative(abi::get_pri(task, out.as_mut_ptr()))?;
        Ok(out.assume_init())
    }
}
