extern "C" fn _f() -> libc::uintptr_t {}
//~^ ERROR cannot find item `libc`

fn main() {}
