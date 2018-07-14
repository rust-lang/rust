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
mod floor;
mod floorf;
mod fmodf;
mod hypot;
mod hypotf;
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
    ceilf::ceilf, expf::expf, fabs::fabs, fabsf::fabsf, floor::floor, floorf::floorf, fmodf::fmodf,
    hypot::hypot, hypotf::hypotf, log2::log2, log2f::log2f, logf::logf, powf::powf, round::round,
    roundf::roundf, scalbn::scalbn, scalbnf::scalbnf, sqrt::sqrt, sqrtf::sqrtf, trunc::trunc,
    truncf::truncf,
};

fn isnanf(x: f32) -> bool {
    x.to_bits() & 0x7fffffff > 0x7f800000
}
