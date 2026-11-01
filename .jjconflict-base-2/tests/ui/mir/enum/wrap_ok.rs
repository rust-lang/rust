//@ run-pass
//@ compile-flags: -C debug-assertions

#[allow(dead_code)]
enum Wrap {
    A(u32),
}

fn main() {
    let _val: Wrap = unsafe { std::mem::transmute::<u32, Wrap>(2) };
    let _val: Wrap = unsafe { std::mem::transmute::<u32, Wrap>(u32::MAX) };
}
