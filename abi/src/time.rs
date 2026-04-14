//! Shared time ABI for kernel and userspace.

/// Clock domains supported by [`crate::syscall::SYS_TIME_NOW`].
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockId {
    Monotonic = 1,
    Realtime = 2,
}

impl ClockId {
    pub const fn from_u32(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::Monotonic),
            2 => Some(Self::Realtime),
            _ => None,
        }
    }
}

/// A 64-bit timespec-like value used by time syscalls.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TimeSpec {
    pub secs: u64,
    pub nanos: u32,
    pub reserved: u32,
}

impl TimeSpec {
    pub const ZERO: Self = Self {
        secs: 0,
        nanos: 0,
        reserved: 0,
    };

    pub const fn is_valid(&self) -> bool {
        self.nanos < 1_000_000_000
    }

    pub const fn from_nanos(total_nanos: u64) -> Self {
        Self {
            secs: total_nanos / 1_000_000_000,
            nanos: (total_nanos % 1_000_000_000) as u32,
            reserved: 0,
        }
    }

    pub const fn as_nanos(&self) -> Option<u64> {
        if !self.is_valid() {
            return None;
        }
        let secs_ns = match self.secs.checked_mul(1_000_000_000) {
            Some(v) => v,
            None => return None,
        };
        secs_ns.checked_add(self.nanos as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::{ClockId, TimeSpec};

    #[test]
    fn clock_id_from_u32_rejects_unknown_values() {
        assert_eq!(ClockId::from_u32(1), Some(ClockId::Monotonic));
        assert_eq!(ClockId::from_u32(2), Some(ClockId::Realtime));
        assert_eq!(ClockId::from_u32(0), None);
        assert_eq!(ClockId::from_u32(99), None);
    }

    #[test]
    fn timespec_round_trips_nanos() {
        let spec = TimeSpec::from_nanos(12_345_678_901);
        assert_eq!(spec.secs, 12);
        assert_eq!(spec.nanos, 345_678_901);
        assert_eq!(spec.as_nanos(), Some(12_345_678_901));
    }

    #[test]
    fn timespec_rejects_invalid_nanoseconds_field() {
        let spec = TimeSpec {
            secs: 1,
            nanos: 1_000_000_000,
            reserved: 0,
        };
        assert!(!spec.is_valid());
        assert_eq!(spec.as_nanos(), None);
    }
}
