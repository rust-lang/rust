// https://github.com/rust-lang/rust/issues/76042
//@ run-pass
//@ compile-flags: -Coverflow-checks=off -Ccodegen-units=1 -Copt-level=0

fn foo(a: i128, b: i128, s: u32) -> (i128, i128) {
    if s == 128 {
        (0, 0)
    } else {
        (b >> s, a >> s)
    }
}
fn main() {
    let r = foo(0, 8, 1);
    if r.0 != 4 {
        panic!();
    }
}
