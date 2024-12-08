extern "C" fn _f() -> libc::uintptr_t {}
//~^ ERROR failed to resolve: use of undeclared crate or module `libc`

fn main() {}
