// error-pattern:aFdEfSeVEE
// compile-flags: -C linker-flavor=ld

/* We're testing that link_args are indeed passed when nolink is specified.
So we try to compile with junk link_args and make sure they are visible in
the compiler output. */

#![feature(link_args)]

#[link_args = "aFdEfSeVEEE"]
extern {}

fn main() { }
