pub static mut A: u32 = 0;
pub static mut B: () = unsafe { A = 1; };
//~^ ERROR could not evaluate static initializer

pub static mut C: u32 = unsafe { C = 1; 0 };
//~^ ERROR cycle detected

pub static D: u32 = D;

fn main() {}
