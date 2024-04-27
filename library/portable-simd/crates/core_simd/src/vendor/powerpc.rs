use crate::simd::*;

#[cfg(target_arch = "powerpc")]
use core::arch::powerpc::*;

#[cfg(target_arch = "powerpc64")]
use core::arch::powerpc64::*;

from_transmute! { unsafe f64x2 => vector_double }
from_transmute! { unsafe i64x2 => vector_signed_long }
from_transmute! { unsafe u64x2 => vector_unsigned_long }
