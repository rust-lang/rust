// edition:2018
// compile-flags: -Z mir-opt-level=2 -Z unsound-mir-opts

#[inline(always)]
pub fn f(s: bool) -> String {
    let a = "Hello world!".to_string();
    let b = a;
    let c = b;
    if s {
        c
    } else {
        String::new()
    }
}
