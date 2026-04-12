//! Provide helper functions/trait impls to pack/unpack
//! [`SyscallArgs`].
//!
//! `io:Error` is not implemented for better `no_std` support.

/// The 6 arguments of a syscall, raw untyped version.
#[derive(PartialEq, Debug, Eq, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SyscallArgs {
    pub arg0: usize,
    pub arg1: usize,
    pub arg2: usize,
    pub arg3: usize,
    pub arg4: usize,
    pub arg5: usize,
}

impl SyscallArgs {
    pub fn new(
        a0: usize,
        a1: usize,
        a2: usize,
        a3: usize,
        a4: usize,
        a5: usize,
    ) -> Self {
        SyscallArgs {
            arg0: a0,
            arg1: a1,
            arg2: a2,
            arg3: a3,
            arg4: a4,
            arg5: a5,
        }
    }
}

impl From<&[usize; 6]> for SyscallArgs {
    fn from(args: &[usize; 6]) -> Self {
        SyscallArgs {
            arg0: args[0],
            arg1: args[1],
            arg2: args[2],
            arg3: args[3],
            arg4: args[4],
            arg5: args[5],
        }
    }
}

impl From<&[usize; 5]> for SyscallArgs {
    fn from(args: &[usize; 5]) -> Self {
        SyscallArgs {
            arg0: args[0],
            arg1: args[1],
            arg2: args[2],
            arg3: args[3],
            arg4: args[4],
            arg5: 0,
        }
    }
}

impl From<&[usize; 4]> for SyscallArgs {
    fn from(args: &[usize; 4]) -> Self {
        SyscallArgs {
            arg0: args[0],
            arg1: args[1],
            arg2: args[2],
            arg3: args[3],
            arg4: 0,
            arg5: 0,
        }
    }
}

impl From<&[usize; 3]> for SyscallArgs {
    fn from(args: &[usize; 3]) -> Self {
        SyscallArgs {
            arg0: args[0],
            arg1: args[1],
            arg2: args[2],
            arg3: 0,
            arg4: 0,
            arg5: 0,
        }
    }
}

impl From<&[usize; 2]> for SyscallArgs {
    fn from(args: &[usize; 2]) -> Self {
        SyscallArgs {
            arg0: args[0],
            arg1: args[1],
            arg2: 0,
            arg3: 0,
            arg4: 0,
            arg5: 0,
        }
    }
}

impl From<&[usize; 1]> for SyscallArgs {
    fn from(args: &[usize; 1]) -> Self {
        SyscallArgs {
            arg0: args[0],
            arg1: 0,
            arg2: 0,
            arg3: 0,
            arg4: 0,
            arg5: 0,
        }
    }
}

impl From<&[usize; 0]> for SyscallArgs {
    fn from(_args: &[usize; 0]) -> Self {
        SyscallArgs {
            arg0: 0,
            arg1: 0,
            arg2: 0,
            arg3: 0,
            arg4: 0,
            arg5: 0,
        }
    }
}

#[macro_export]
macro_rules! syscall_args {
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr) => {
        $crate::SyscallArgs::new($a, $b, $c, $d, $e, $f)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {
        $crate::SyscallArgs::new($a, $b, $c, $d, $e, 0)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        $crate::SyscallArgs::new($a, $b, $c, $d, 0, 0)
    };
    ($a:expr, $b:expr, $c:expr) => {
        $crate::SyscallArgs::new($a, $b, $c, 0, 0, 0)
    };
    ($a:expr, $b:expr) => {
        $crate::SyscallArgs::new($a, $b, 0, 0, 0, 0)
    };
    ($a:expr) => {
        $crate::SyscallArgs::new($a, 0, 0, 0, 0, 0)
    };
    () => {
        $crate::SyscallArgs::new(0, 0, 0, 0, 0, 0)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syscall_args_macro_test() {
        assert_eq!(
            syscall_args!(1, 2, 3, 4, 5, 6),
            SyscallArgs::new(1, 2, 3, 4, 5, 6)
        );
        assert_eq!(
            syscall_args!(1, 2, 3, 4, 5),
            SyscallArgs::new(1, 2, 3, 4, 5, 0)
        );
        assert_eq!(
            syscall_args!(1, 2, 3, 4),
            SyscallArgs::new(1, 2, 3, 4, 0, 0)
        );
        assert_eq!(syscall_args!(1, 2, 3), SyscallArgs::new(1, 2, 3, 0, 0, 0));
        assert_eq!(syscall_args!(1, 2), SyscallArgs::new(1, 2, 0, 0, 0, 0));
        assert_eq!(syscall_args!(1), SyscallArgs::new(1, 0, 0, 0, 0, 0));
        assert_eq!(syscall_args!(), SyscallArgs::new(0, 0, 0, 0, 0, 0));
    }

    #[test]
    fn syscall_args_from_u64_slice() {
        assert_eq!(
            SyscallArgs::from(&[1, 2, 3, 4, 5, 6]),
            syscall_args!(1, 2, 3, 4, 5, 6)
        );
        assert_eq!(
            SyscallArgs::from(&[1, 2, 3, 4, 5]),
            syscall_args!(1, 2, 3, 4, 5)
        );
        assert_eq!(SyscallArgs::from(&[1, 2, 3, 4]), syscall_args!(1, 2, 3, 4));
        assert_eq!(SyscallArgs::from(&[1, 2, 3]), syscall_args!(1, 2, 3));
        assert_eq!(SyscallArgs::from(&[1, 2]), syscall_args!(1, 2));
        assert_eq!(SyscallArgs::from(&[1]), syscall_args!(1));
        assert_eq!(SyscallArgs::from(&[0]), syscall_args!());
    }
}
