use super::abi;
use super::error::expect_success;
use crate::mem::MaybeUninit;
use crate::time::Duration;

#[cfg(test)]
mod tests;

#[inline]
pub fn get_tim() -> abi::SYSTIM {
    // Safety: The provided pointer is valid
    unsafe {
        let mut out = MaybeUninit::uninit();
        expect_success(abi::get_tim(out.as_mut_ptr()), &"get_tim");
        out.assume_init()
    }
}

/// Split `Duration` into zero or more `RELTIM`s.
#[inline]
pub fn dur2reltims(dur: Duration) -> impl Iterator<Item = abi::RELTIM> {
    // `RELTIM` is microseconds
    let mut ticks = dur.as_micros();

    crate::iter::from_fn(move || {
        if ticks == 0 {
            None
        } else if ticks <= abi::TMAX_RELTIM as u128 {
            Some(crate::mem::replace(&mut ticks, 0) as abi::RELTIM)
        } else {
            ticks -= abi::TMAX_RELTIM as u128;
            Some(abi::TMAX_RELTIM)
        }
    })
}

/// Split `Duration` into one or more `TMO`s.
#[inline]
fn dur2tmos(dur: Duration) -> impl Iterator<Item = abi::TMO> {
    // `TMO` is microseconds
    let mut ticks = dur.as_micros();
    let mut end = false;

    crate::iter::from_fn(move || {
        if end {
            None
        } else if ticks <= abi::TMAX_RELTIM as u128 {
            end = true;
            Some(crate::mem::replace(&mut ticks, 0) as abi::TMO)
        } else {
            ticks -= abi::TMAX_RELTIM as u128;
            Some(abi::TMAX_RELTIM)
        }
    })
}

/// Split `Duration` into one or more API calls with timeout.
#[inline]
pub fn with_tmos(dur: Duration, mut f: impl FnMut(abi::TMO) -> abi::ER) -> abi::ER {
    let mut er = abi::E_TMOUT;
    for tmo in dur2tmos(dur) {
        er = f(tmo);
        if er != abi::E_TMOUT {
            break;
        }
    }
    er
}

/// Split `Duration` into one or more API calls with timeout. This function can
/// handle spurious wakeups.
#[inline]
pub fn with_tmos_strong(dur: Duration, mut f: impl FnMut(abi::TMO) -> abi::ER) -> abi::ER {
    // `TMO` and `SYSTIM` are microseconds.
    // Clamp at `SYSTIM::MAX` for performance reasons. This shouldn't cause
    // a problem in practice. (`u64::MAX` μs ≈ 584942 years)
    let ticks = dur.as_micros().min(abi::SYSTIM::MAX as u128) as abi::SYSTIM;

    let start = get_tim();
    let mut elapsed = 0;
    let mut er = abi::E_TMOUT;
    while elapsed <= ticks {
        er = f(elapsed.min(abi::TMAX_RELTIM as abi::SYSTIM) as abi::TMO);
        if er != abi::E_TMOUT {
            break;
        }
        elapsed = get_tim().wrapping_sub(start);
    }

    er
}
