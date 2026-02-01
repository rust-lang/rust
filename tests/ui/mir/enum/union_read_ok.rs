//@ run-pass
//@ compile-flags: -C debug-assertions

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
    let foo = Foo { a: unsafe { std::mem::transmute(0_u16) } };

    let val: Single = unsafe { foo.a };
    println!("{}", val as u16);
}
