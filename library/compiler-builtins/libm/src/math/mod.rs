macro_rules! force_eval {
    ($e:expr) => {
        unsafe { ::core::ptr::read_volatile(&$e); }
    }
}

mod fabs;
mod fabsf;
mod fmodf;
mod powf;
mod round;
mod scalbn;
mod scalbnf;
mod sqrtf;
mod logf;
mod expf;
mod floor;
mod cosf;
mod trunc;
mod truncf;

pub use self::{
    fabs::fabs,
    fabsf::fabsf,
    fmodf::fmodf,
    powf::powf,
    round::round,
    scalbn::scalbn,
    scalbnf::scalbnf,
    sqrtf::sqrtf,
    logf::logf,
    expf::expf,
    floor::floor,
    cosf::cosf,
    trunc::trunc,
    truncf::truncf,
};

mod k_cosf;
mod k_sinf;
mod rem_pio2f;
mod rem_pio2_large;

use self::{
    k_cosf::k_cosf,
    k_sinf::k_sinf,
    rem_pio2f::rem_pio2f,
    rem_pio2_large::rem_pio2_large,
};

fn isnanf(x: f32) -> bool {
    x.to_bits() & 0x7fffffff > 0x7f800000
}
