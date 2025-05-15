//@ run-fail
//@ compile-flags: -C debug-assertions
//@ error-pattern: trying to construct an enum from an invalid value 0x1

#[allow(dead_code)]
#[repr(u16)]
enum Single {
    A
}

fn main() {
    let _val: Single = unsafe { std::mem::transmute::<u16, Single>(1) };
}
