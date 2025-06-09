pub static mut A: u32 = 0;
pub static mut B: () = unsafe { A = 1; };
//~^ ERROR modifying a static's initial value

pub static mut C: u32 = unsafe { C = 1; 0 };

pub static D: u32 = D;
//~^ ERROR static that tried to initialize itself with itself

fn main() {}
