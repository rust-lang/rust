struct Foo {
    a: u8,
    b: (),
    c: &'static str,
    d: Option<isize>,
}

// EMIT_MIR flatten_locals.main.FlattenLocals.diff
fn main() {
    let Foo { a, b, c, d } = Foo { a: 5, b: (), c: "a", d: Some(-4) };
    let _ = a;
    let _ = b;
    let _ = c;
    let _ = d;

    // Verify this struct is not flattened.
    f(&S { a: 1, b: 2, c: g() }.a);
}

#[repr(C)]
struct S {
    a: u32,
    b: u32,
    c: u32,
}

fn f(a: *const u32) {
    println!("{}", unsafe { *a.add(2) });
}

fn g() -> u32 {
    3
}
