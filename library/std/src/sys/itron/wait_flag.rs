use crate::mem::MaybeUninit;
use crate::time::Duration;

use super::{
    abi,
    error::{expect_success, fail},
    time::with_tmos,
};

const CLEAR: abi::FLGPTN = 0;
const RAISED: abi::FLGPTN = 1;

/// A thread parking primitive that is not susceptible to race conditions,
/// but provides no atomic ordering guarantees and allows only one `raise` per wait.
pub struct WaitFlag {
    flag: abi::ID,
}

impl WaitFlag {
    /// Creates a new wait flag.
    pub fn new() -> WaitFlag {
        let flag = expect_success(
            unsafe {
                abi::acre_flg(&abi::T_CFLG {
                    flgatr: abi::TA_FIFO | abi::TA_WSGL | abi::TA_CLR,
                    iflgptn: CLEAR,
                })
            },
            &"acre_flg",
        );

        WaitFlag { flag }
    }

    /// Wait for the wait flag to be raised.
    pub fn wait(&self) {
        let mut token = MaybeUninit::uninit();
        expect_success(
            unsafe { abi::wai_flg(self.flag, RAISED, abi::TWF_ORW, token.as_mut_ptr()) },
            &"wai_flg",
        );
    }

    /// Wait for the wait flag to be raised or the timeout to occur.
    ///
    /// Returns whether the flag was raised (`true`) or the operation timed out (`false`).
    pub fn wait_timeout(&self, dur: Duration) -> bool {
        let mut token = MaybeUninit::uninit();
        let res = with_tmos(dur, |tmout| unsafe {
            abi::twai_flg(self.flag, RAISED, abi::TWF_ORW, token.as_mut_ptr(), tmout)
        });

        match res {
            abi::E_OK => true,
            abi::E_TMOUT => false,
            error => fail(error, &"twai_flg"),
        }
    }

    /// Raise the wait flag.
    ///
    /// Calls to this function should be balanced with the number of successful waits.
    pub fn raise(&self) {
        expect_success(unsafe { abi::set_flg(self.flag, RAISED) }, &"set_flg");
    }
}

impl Drop for WaitFlag {
    fn drop(&mut self) {
        expect_success(unsafe { abi::del_flg(self.flag) }, &"del_flg");
    }
}
