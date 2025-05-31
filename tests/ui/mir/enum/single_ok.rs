//@ run-pass
//@ compile-flags: -C debug-assertions

#[allow(dead_code)]
enum Single {
    A
}

fn main() {
    let _val: Single = unsafe { std::mem::transmute::<(), Single>(()) };
}
