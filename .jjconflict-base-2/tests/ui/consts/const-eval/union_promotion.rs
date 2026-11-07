#[repr(C)]
union Foo {
    a: &'static u32,
    b: usize,
}

fn main() {
    let x: &'static bool = &unsafe { //~ ERROR temporary value dropped while borrowed
        Foo { a: &1 }.b == Foo { a: &2 }.b
    };
}
