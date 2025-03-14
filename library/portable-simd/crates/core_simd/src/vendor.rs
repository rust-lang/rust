/// Provides implementations of `From<$a> for $b` and `From<$b> for $a` that transmutes the value.
#[allow(unused)]
macro_rules! from_transmute {
    { unsafe $a:ty => $b:ty } => {
        from_transmute!{ @impl $a => $b }
        from_transmute!{ @impl $b => $a }
    };
    { @impl $from:ty => $to:ty } => {
        impl core::convert::From<$from> for $to {
            #[inline]
            fn from(value: $from) -> $to {
                // Safety: transmuting between vectors is safe, but the caller of this macro
                // checks the invariants
                unsafe { core::mem::transmute(value) }
            }
        }
    };
}

/// Conversions to x86's SIMD types.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86;

#[cfg(target_arch = "wasm32")]
mod wasm32;

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec", target_arch = "arm",))]
mod arm;

#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
mod powerpc;

#[cfg(target_arch = "loongarch64")]
mod loongarch64;
