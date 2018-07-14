macro_rules! force_eval {
    ($e:expr) => {
        unsafe {
            ::core::ptr::read_volatile(&$e);
        }
    };
}

mod ceilf;
mod expf;
mod fabs;
mod fabsf;
mod fdimf;
mod floor;
mod floorf;
mod fmodf;
mod hypot;
mod hypotf;
mod log;
mod log10;
mod log10f;
mod log1p;
mod log1pf;
mod log2;
mod log2f;
mod logf;
mod powf;
mod round;
mod roundf;
mod scalbn;
mod scalbnf;
mod sqrt;
mod sqrtf;
mod trunc;
mod truncf;

//mod service;

pub use self::{
    ceilf::ceilf, expf::expf, fabs::fabs, fabsf::fabsf, fdimf::fdimf, floor::floor, floorf::floorf,
    fmodf::fmodf, hypot::hypot, hypotf::hypotf, log::log, log10::log10, log10f::log10f,
    log1p::log1p, log1pf::log1pf, log2::log2, log2f::log2f, logf::logf, powf::powf, round::round,
    roundf::roundf, scalbn::scalbn, scalbnf::scalbnf, sqrt::sqrt, sqrtf::sqrtf, trunc::trunc,
    truncf::truncf,
};

fn isnanf(x: f32) -> bool {
    x.to_bits() & 0x7fffffff > 0x7f800000
}
