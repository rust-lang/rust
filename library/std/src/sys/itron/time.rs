use super::{abi, error::expect_success};
use crate::{convert::TryInto, mem::MaybeUninit, time::Duration};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Instant(abi::SYSTIM);

impl Instant {
    pub fn now() -> Instant {
        // Safety: The provided pointer is valid
        unsafe {
            let mut out = MaybeUninit::uninit();
            expect_success(abi::get_tim(out.as_mut_ptr()), &"get_tim");
            Instant(out.assume_init())
        }
    }

    pub const fn zero() -> Instant {
        Instant(0)
    }

    pub fn actually_monotonic() -> bool {
        // There are ways to change the system time
        false
    }

    pub fn checked_sub_instant(&self, other: &Instant) -> Option<Duration> {
        self.0.checked_sub(other.0).map(|ticks| {
            // `SYSTIM` is measured in microseconds
            Duration::from_micros(ticks)
        })
    }

    pub fn checked_add_duration(&self, other: &Duration) -> Option<Instant> {
        // `SYSTIM` is measured in microseconds
        let ticks = other.as_micros();

        Some(Instant(self.0.checked_add(ticks.try_into().ok()?)?))
    }

    pub fn checked_sub_duration(&self, other: &Duration) -> Option<Instant> {
        // `SYSTIM` is measured in microseconds
        let ticks = other.as_micros();

        Some(Instant(self.0.checked_sub(ticks.try_into().ok()?)?))
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

    let start = Instant::now().0;
    let mut elapsed = 0;
    let mut er = abi::E_TMOUT;
    while elapsed <= ticks {
        er = f(elapsed.min(abi::TMAX_RELTIM as abi::SYSTIM) as abi::TMO);
        if er != abi::E_TMOUT {
            break;
        }
        elapsed = Instant::now().0.wrapping_sub(start);
    }

    er
}

#[cfg(test)]
mod tests;
