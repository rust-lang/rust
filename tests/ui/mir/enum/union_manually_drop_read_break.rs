//@ run-crash
//@ compile-flags: -C debug-assertions
//@ error-pattern: trying to construct an enum from an invalid value 0x1

#[allow(dead_code)]
#[repr(u16)]
enum Single {
    A,
}

union Foo {
    a: std::mem::ManuallyDrop<Single>,
}

fn main() {
    let foo = Foo { a: unsafe { std::mem::transmute(1_u16) } };

    let val: Single = unsafe { std::mem::ManuallyDrop::into_inner(foo.a) };
    println!("{}", val as u16);
}
