// run-rustfix
// compile-flags: -Z continue-parse-after-error

extern "C" {
    const C: u8; //~ ERROR extern items cannot be `const`
}

fn main() {
    // We suggest turning the (illegal) extern `const` into an extern `static`,
    // but this also requires `unsafe` (a deny-by-default lint at comment time,
    // future error; Issue #36247)
    unsafe {
        let _x = C;
    }
}
