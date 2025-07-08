//@ run-fail
//@ compile-flags: -C debug-assertions
//@ check-run-results

#[allow(dead_code)]
#[repr(u16)]
enum Single {
    A
}

fn main() {
    let _val: Single = unsafe { std::mem::transmute::<u16, Single>(1) };
}
