/// Provides implementations of `From<$a> for $b` and `From<$b> for $a` that transmutes the value.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! from_transmute {
    { unsafe $a:ty => $b:ty } => {
        from_transmute!{ @impl $a => $b }
        from_transmute!{ @impl $b => $a }
    };
    { @impl $from:ty => $to:ty } => {
        impl core::convert::From<$from> for $to {
            #[inline]
            fn from(value: $from) -> $to {
                unsafe { core::mem::transmute(value) }
            }
        }
    };
}

/// Provides implementations of `From<$generic> for core::arch::{x86, x86_64}::$intel` and
/// vice-versa that transmutes the value.
macro_rules! from_transmute_x86 {
    { unsafe $generic:ty => $intel:ident } => {
        #[cfg(target_arch = "x86")]
        from_transmute! { unsafe $generic => core::arch::x86::$intel }

        #[cfg(target_arch = "x86_64")]
        from_transmute! { unsafe $generic => core::arch::x86_64::$intel }
    }
}
