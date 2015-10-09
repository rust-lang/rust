pub use imp::c as imp;

pub mod prelude {
    pub use super::imp::{
        c_int, c_float, c_double, c_char,
        EINVAL, EIO,
        strlen
    };
}
