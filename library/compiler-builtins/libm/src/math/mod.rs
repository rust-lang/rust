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
mod fmod;
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
    ceilf::ceilf, expf::expf, fabs::fabs, fabsf::fabsf, floor::floor, floorf::floorf, fmod::fmod,
    fmodf::fmodf, hypot::hypot, hypotf::hypotf, logf::logf, powf::powf, round::round,
    roundf::roundf, scalbn::scalbn, scalbnf::scalbnf, sqrt::sqrt, sqrtf::sqrtf, trunc::trunc,
    truncf::truncf,
};
