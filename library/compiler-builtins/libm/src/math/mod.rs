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
mod scalbnf;
mod sqrtf;
mod logf;
mod expf;

pub use self::fabs::fabs;
pub use self::fabsf::fabsf;
pub use self::fmodf::fmodf;
pub use self::powf::powf;
pub use self::round::round;
pub use self::scalbnf::scalbnf;
pub use self::sqrtf::sqrtf;
pub use self::logf::logf;
pub use self::expf::expf;

fn isnanf(x: f32) -> bool {
    x.to_bits() & 0x7fffffff > 0x7f800000
}
