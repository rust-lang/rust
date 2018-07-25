#![feature(lang_items)]
#![feature(panic_implementation)]
#![no_main]
#![no_std]

extern crate libm;

use core::panic::PanicInfo;
use core::ptr;

macro_rules! force_eval {
    ($e:expr) => {
        unsafe {
            core::ptr::read_volatile(&$e);
        }
    };
}

#[no_mangle]
pub fn main() {
    force_eval!(libm::acos(random()));
    force_eval!(libm::acosf(random()));
    force_eval!(libm::asin(random()));
    force_eval!(libm::asinf(random()));
    force_eval!(libm::atan(random()));
    force_eval!(libm::atan2(random(), random()));
    force_eval!(libm::atan2f(random(), random()));
    force_eval!(libm::atanf(random()));
    force_eval!(libm::cbrt(random()));
    force_eval!(libm::cbrtf(random()));
    force_eval!(libm::ceil(random()));
    force_eval!(libm::ceilf(random()));
    force_eval!(libm::cos(random()));
    force_eval!(libm::cosf(random()));
    force_eval!(libm::cosh(random()));
    force_eval!(libm::coshf(random()));
    force_eval!(libm::exp(random()));
    force_eval!(libm::exp2(random()));
    force_eval!(libm::exp2f(random()));
    force_eval!(libm::expf(random()));
    force_eval!(libm::expm1(random()));
    force_eval!(libm::expm1f(random()));
    force_eval!(libm::fabs(random()));
    force_eval!(libm::fabsf(random()));
    force_eval!(libm::fdim(random(), random()));
    force_eval!(libm::fdimf(random(), random()));
    force_eval!(libm::floor(random()));
    force_eval!(libm::floorf(random()));
    force_eval!(libm::fma(random(), random(), random()));
    force_eval!(libm::fmaf(random(), random(), random()));
    force_eval!(libm::fmod(random(), random()));
    force_eval!(libm::fmodf(random(), random()));
    force_eval!(libm::hypot(random(), random()));
    force_eval!(libm::hypotf(random(), random()));
    force_eval!(libm::log(random()));
    force_eval!(libm::log2(random()));
    force_eval!(libm::log10(random()));
    force_eval!(libm::log10f(random()));
    force_eval!(libm::log1p(random()));
    force_eval!(libm::log1pf(random()));
    force_eval!(libm::log2f(random()));
    force_eval!(libm::logf(random()));
    force_eval!(libm::pow(random(), random()));
    force_eval!(libm::powf(random(), random()));
    force_eval!(libm::round(random()));
    force_eval!(libm::roundf(random()));
    force_eval!(libm::scalbn(random(), random()));
    force_eval!(libm::scalbnf(random(), random()));
    force_eval!(libm::sin(random()));
    force_eval!(libm::sinf(random()));
    force_eval!(libm::sinh(random()));
    force_eval!(libm::sinhf(random()));
    force_eval!(libm::sqrt(random()));
    force_eval!(libm::sqrtf(random()));
    force_eval!(libm::tan(random()));
    force_eval!(libm::tanf(random()));
    force_eval!(libm::tanh(random()));
    force_eval!(libm::tanhf(random()));
    force_eval!(libm::trunc(random()));
    force_eval!(libm::truncf(random()));
}

fn random<T>() -> T
where
    T: Copy,
{
    unsafe {
        static mut X: usize = 0;
        X += 8;
        ptr::read_volatile(X as *const T)
    }
}

#[panic_implementation]
#[no_mangle]
pub fn panic(_info: &PanicInfo) -> ! {
    // loop {}
    extern "C" {
        fn thou_shalt_not_panic() -> !;
    }

    unsafe { thou_shalt_not_panic() }
}

#[link(name = "c")]
extern "C" {}

#[lang = "eh_personality"]
fn eh() {}

#[no_mangle]
pub extern "C" fn __aeabi_unwind_cpp_pr0() {}

#[no_mangle]
pub extern "C" fn __aeabi_unwind_cpp_pr1() {}
