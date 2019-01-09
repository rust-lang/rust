// Constants (static variables) can be used to match in patterns, but mutable
// statics cannot. This ensures that there's some form of error if this is
// attempted.

extern crate libc;

extern {
    static mut rust_dbg_static_mut: libc::c_int;
    pub fn rust_dbg_static_mut_check_four();
    #[cfg(stage37)] //~ ERROR expected item after attributes
}

pub fn main() {}
