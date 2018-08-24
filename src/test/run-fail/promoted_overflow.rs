#![allow(const_err)]

// error-pattern: overflow
// compile-flags: -C overflow-checks=yes

fn main() {
    let x: &'static u32 = &(0u32 - 1);
}
