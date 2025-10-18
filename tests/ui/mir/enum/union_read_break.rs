//@ run-crash
//@ compile-flags: -C debug-assertions
//@ error-pattern: trying to construct an enum from an invalid value 0x1

#[allow(dead_code)]
#[repr(u16)]
#[derive(Copy, Clone)]
enum Single {
    A,
}

union Foo {
    a: Single,
}

fn main() {
    let foo = Foo { a: unsafe { std::mem::transmute(1_u16) } };

    let val: Single = unsafe { foo.a };
    println!("{}", val as u16);
}
