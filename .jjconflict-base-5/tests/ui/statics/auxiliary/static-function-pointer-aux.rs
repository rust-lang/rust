pub fn f(x: isize) -> isize { -x }

pub static F: fn(isize) -> isize = f;
pub static mut MutF: fn(isize) -> isize = f;
