//@ run-pass
//@ compile-flags: -C debug-assertions

#[allow(dead_code)]
#[repr(u16)]
enum Single {
    A
}

fn main() {
    let _val: Single = unsafe { std::mem::transmute::<u16, Single>(0) };
}
