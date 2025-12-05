use super::abi;
use super::error::expect_success_aborting;
use super::time::with_tmos;
use crate::time::Duration;

pub type ThreadId = abi::ID;

pub use super::task::current_task_id_aborting as current;

pub fn park(_hint: usize) {
    match unsafe { abi::slp_tsk() } {
        abi::E_OK | abi::E_RLWAI => {}
        err => {
            expect_success_aborting(err, &"slp_tsk");
        }
    }
}

pub fn park_timeout(dur: Duration, _hint: usize) {
    match with_tmos(dur, |tmo| unsafe { abi::tslp_tsk(tmo) }) {
        abi::E_OK | abi::E_RLWAI | abi::E_TMOUT => {}
        err => {
            expect_success_aborting(err, &"tslp_tsk");
        }
    }
}

pub fn unpark(id: ThreadId, _hint: usize) {
    match unsafe { abi::wup_tsk(id) } {
        // It is allowed to try to wake up a destroyed or unrelated task, so we ignore all
        // errors that could result from that situation.
        abi::E_OK | abi::E_NOEXS | abi::E_OBJ | abi::E_QOVR => {}
        err => {
            expect_success_aborting(err, &"wup_tsk");
        }
    }
}
