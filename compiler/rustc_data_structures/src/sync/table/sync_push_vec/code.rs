#![cfg(code)]

use super::{SyncPushVec, Write};

#[no_mangle]
unsafe fn get(a: &SyncPushVec<usize>) -> Option<usize> {
    a.unsafe_write().read().as_slice().get(2).cloned()
}

#[no_mangle]
unsafe fn push(a: &SyncPushVec<usize>) {
    a.unsafe_write().push(4000);
}

#[no_mangle]
unsafe fn push2(a: &mut Write<'_, usize>) {
    a.push(4000);
}
