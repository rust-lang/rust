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
mod scalbnf;
mod sqrtf;

pub use self::fabs::fabs;
pub use self::fabsf::fabsf;
pub use self::fmodf::fmodf;
pub use self::powf::powf;
pub use self::round::round;
pub use self::roundf::roundf;
pub use self::scalbnf::scalbnf;
pub use self::sqrtf::sqrtf;

fn isnanf(x: f32) -> bool {
    x.to_bits() & 0x7fffffff > 0x7f800000
}
