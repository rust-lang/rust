macro_rules! force_eval {
    ($e:expr) => {
        unsafe {
            ::core::ptr::read_volatile(&$e);
        }
    };
}

mod fabs;
mod fabsf;
mod fmodf;
mod powf;
mod round;
mod roundf;
mod scalbn;
mod scalbnf;
mod sqrt;
mod sqrtf;
mod logf;
mod expf;
mod floor;
mod trunc;
mod truncf;
mod hypot;
mod hypotf;

//mod service;

pub use self::{
    fabs::fabs,
    fabsf::fabsf,
    fmodf::fmodf,
    powf::powf,
    round::round,
    roundf::roundf,
    scalbn::scalbn,
    scalbnf::scalbnf,
    sqrt::sqrt,
    sqrtf::sqrtf,
    logf::logf,
    expf::expf,
    floor::floor,
    trunc::trunc,
    truncf::truncf,
    hypot::hypot,
    hypotf::hypotf,
};

fn isnanf(x: f32) -> bool {
    x.to_bits() & 0x7fffffff > 0x7f800000
}
