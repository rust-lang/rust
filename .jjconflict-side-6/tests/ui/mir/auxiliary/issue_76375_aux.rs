//@ edition:2018
//@ compile-flags: -Z mir-opt-level=3

#[inline(always)]
pub fn copy_prop(s: bool) -> String {
    let a = "Hello world!".to_string();
    let b = a;
    let c = b;
    if s {
        c
    } else {
        String::new()
    }
}

#[inline(always)]
pub fn dest_prop(x: &[u8]) -> &[u8] {
    let y = &x[..x.len()];
    y
}
