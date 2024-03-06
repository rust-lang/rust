#![allow(unused)]

#[unstable(feature = "sgx_platform", issue = "56975")]
pub use fortanix_sgx_abi::*;

use crate::num::NonZero;
use crate::ptr::NonNull;

#[repr(C)]
struct UsercallReturn(u64, u64);

extern "C" {
    fn usercall(nr: NonZero<u64>, p1: u64, p2: u64, abort: u64, p3: u64, p4: u64)
    -> UsercallReturn;
}

/// Performs the raw usercall operation as defined in the ABI calling convention.
///
/// # Safety
///
/// The caller must ensure to pass parameters appropriate for the usercall `nr`
/// and to observe all requirements specified in the ABI.
///
/// # Panics
///
/// Panics if `nr` is `0`.
#[unstable(feature = "sgx_platform", issue = "56975")]
#[inline]
pub unsafe fn do_usercall(
    nr: NonZero<u64>,
    p1: u64,
    p2: u64,
    p3: u64,
    p4: u64,
    abort: bool,
) -> (u64, u64) {
    let UsercallReturn(a, b) = unsafe { usercall(nr, p1, p2, abort as _, p3, p4) };
    (a, b)
}

/// A value passed or returned in a CPU register.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub type Register = u64;

/// Translate a type from/to Register to be used as an argument.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub trait RegisterArgument {
    /// Translate a Register to Self.
    fn from_register(_: Register) -> Self;
    /// Translate self to a Register.
    fn into_register(self) -> Register;
}

/// Translate a pair of Registers to the raw usercall return value.
#[unstable(feature = "sgx_platform", issue = "56975")]
pub trait ReturnValue {
    /// Translate a pair of Registers to the raw usercall return value.
    fn from_registers(call: &'static str, regs: (Register, Register)) -> Self;
}

macro_rules! define_usercalls {
    ($(fn $f:ident($($n:ident: $t:ty),*) $(-> $r:tt)*; )*) => {
        /// Usercall numbers as per the ABI.
        #[repr(u64)]
        #[unstable(feature = "sgx_platform", issue = "56975")]
        #[derive(Copy, Clone, Hash, PartialEq, Eq, Debug)]
        #[allow(missing_docs, non_camel_case_types)]
        #[non_exhaustive]
        pub enum Usercalls {
            #[doc(hidden)]
            __enclave_usercalls_invalid = 0,
            $($f,)*
        }

        $(enclave_usercalls_internal_define_usercalls!(def fn $f($($n: $t),*) $(-> $r)*);)*
    };
}

macro_rules! define_ra {
    (< $i:ident > $t:ty) => {
        #[unstable(feature = "sgx_platform", issue = "56975")]
        impl<$i> RegisterArgument for $t {
            fn from_register(a: Register) -> Self {
                a as _
            }
            fn into_register(self) -> Register {
                self as _
            }
        }
    };
    ($i:ty as $t:ty) => {
        #[unstable(feature = "sgx_platform", issue = "56975")]
        impl RegisterArgument for $t {
            fn from_register(a: Register) -> Self {
                a as $i as _
            }
            fn into_register(self) -> Register {
                self as $i as _
            }
        }
    };
    ($t:ty) => {
        #[unstable(feature = "sgx_platform", issue = "56975")]
        impl RegisterArgument for $t {
            fn from_register(a: Register) -> Self {
                a as _
            }
            fn into_register(self) -> Register {
                self as _
            }
        }
    };
}

define_ra!(Register);
define_ra!(i64);
define_ra!(u32);
define_ra!(u32 as i32);
define_ra!(u16);
define_ra!(u16 as i16);
define_ra!(u8);
define_ra!(u8 as i8);
define_ra!(usize);
define_ra!(usize as isize);
define_ra!(<T> *const T);
define_ra!(<T> *mut T);

#[unstable(feature = "sgx_platform", issue = "56975")]
impl RegisterArgument for bool {
    fn from_register(a: Register) -> bool {
        if a != 0 { true } else { false }
    }
    fn into_register(self) -> Register {
        self as _
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T: RegisterArgument> RegisterArgument for Option<NonNull<T>> {
    fn from_register(a: Register) -> Option<NonNull<T>> {
        NonNull::new(a as _)
    }
    fn into_register(self) -> Register {
        self.map_or(0 as _, NonNull::as_ptr) as _
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl ReturnValue for ! {
    fn from_registers(call: &'static str, _regs: (Register, Register)) -> Self {
        rtabort!("Usercall {call}: did not expect to be re-entered");
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl ReturnValue for () {
    fn from_registers(call: &'static str, usercall_retval: (Register, Register)) -> Self {
        rtassert!(usercall_retval.0 == 0);
        rtassert!(usercall_retval.1 == 0);
        ()
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T: RegisterArgument> ReturnValue for T {
    fn from_registers(call: &'static str, usercall_retval: (Register, Register)) -> Self {
        rtassert!(usercall_retval.1 == 0);
        T::from_register(usercall_retval.0)
    }
}

#[unstable(feature = "sgx_platform", issue = "56975")]
impl<T: RegisterArgument, U: RegisterArgument> ReturnValue for (T, U) {
    fn from_registers(_call: &'static str, regs: (Register, Register)) -> Self {
        (T::from_register(regs.0), U::from_register(regs.1))
    }
}

macro_rules! return_type_is_abort {
    (!) => {
        true
    };
    ($r:ty) => {
        false
    };
}

// In this macro: using `$r:tt` because `$r:ty` doesn't match ! in `return_type_is_abort`
macro_rules! enclave_usercalls_internal_define_usercalls {
    (def fn $f:ident($n1:ident: $t1:ty, $n2:ident: $t2:ty,
                     $n3:ident: $t3:ty, $n4:ident: $t4:ty) -> $r:tt) => (
        /// This is the raw function definition, see the ABI documentation for
        /// more information.
        #[unstable(feature = "sgx_platform", issue = "56975")]
        #[inline(always)]
        pub unsafe fn $f($n1: $t1, $n2: $t2, $n3: $t3, $n4: $t4) -> $r {
            ReturnValue::from_registers(stringify!($f), unsafe { do_usercall(
                    rtunwrap!(Some, NonZero::new(Usercalls::$f as Register)),
                    RegisterArgument::into_register($n1),
                    RegisterArgument::into_register($n2),
                    RegisterArgument::into_register($n3),
                    RegisterArgument::into_register($n4),
                    return_type_is_abort!($r)
            ) })
        }
    );
    (def fn $f:ident($n1:ident: $t1:ty, $n2:ident: $t2:ty, $n3:ident: $t3:ty) -> $r:tt) => (
        /// This is the raw function definition, see the ABI documentation for
        /// more information.
        #[unstable(feature = "sgx_platform", issue = "56975")]
        #[inline(always)]
        pub unsafe fn $f($n1: $t1, $n2: $t2, $n3: $t3) -> $r {
            ReturnValue::from_registers(stringify!($f), unsafe { do_usercall(
                    rtunwrap!(Some, NonZero::new(Usercalls::$f as Register)),
                    RegisterArgument::into_register($n1),
                    RegisterArgument::into_register($n2),
                    RegisterArgument::into_register($n3),
                    0,
                    return_type_is_abort!($r)
            ) })
        }
    );
    (def fn $f:ident($n1:ident: $t1:ty, $n2:ident: $t2:ty) -> $r:tt) => (
        /// This is the raw function definition, see the ABI documentation for
        /// more information.
        #[unstable(feature = "sgx_platform", issue = "56975")]
        #[inline(always)]
        pub unsafe fn $f($n1: $t1, $n2: $t2) -> $r {
            ReturnValue::from_registers(stringify!($f), unsafe { do_usercall(
                    rtunwrap!(Some, NonZero::new(Usercalls::$f as Register)),
                    RegisterArgument::into_register($n1),
                    RegisterArgument::into_register($n2),
                    0,0,
                    return_type_is_abort!($r)
            ) })
        }
    );
    (def fn $f:ident($n1:ident: $t1:ty) -> $r:tt) => (
        /// This is the raw function definition, see the ABI documentation for
        /// more information.
        #[unstable(feature = "sgx_platform", issue = "56975")]
        #[inline(always)]
        pub unsafe fn $f($n1: $t1) -> $r {
            ReturnValue::from_registers(stringify!($f), unsafe { do_usercall(
                    rtunwrap!(Some, NonZero::new(Usercalls::$f as Register)),
                    RegisterArgument::into_register($n1),
                    0,0,0,
                    return_type_is_abort!($r)
            ) })
        }
    );
    (def fn $f:ident() -> $r:tt) => (
        /// This is the raw function definition, see the ABI documentation for
        /// more information.
        #[unstable(feature = "sgx_platform", issue = "56975")]
        #[inline(always)]
        pub unsafe fn $f() -> $r {
            ReturnValue::from_registers(stringify!($f), unsafe { do_usercall(
                    rtunwrap!(Some, NonZero::new(Usercalls::$f as Register)),
                    0,0,0,0,
                    return_type_is_abort!($r)
            ) })
        }
    );
    (def fn $f:ident($($n:ident: $t:ty),*)) => (
        enclave_usercalls_internal_define_usercalls!(def fn $f($($n: $t),*) -> ());
    );
}

invoke_with_usercalls!(define_usercalls);
