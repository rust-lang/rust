//@ run-pass
#![allow(unreachable_patterns)]

#[derive(Copy, Clone)]
#[allow(dead_code)]
struct Pie {
    slices: u8,
    size: u8,
}

union Foo {
    #[allow(dead_code)]
    bar: i8,
    baz: Pie,
}

fn main() {
    let u = Foo { bar: 5 };
    let (Some(Foo { bar: _ }) | None) = Some(u);
    let u = Foo { bar: 6 };
    let (Some(Foo { bar: _ }) | Some(Foo { bar: _ }) | None) = Some(u);
    unsafe {
        let u = Foo { bar: 7 };
        let (Foo { bar } | Foo { bar }) = u;
        assert_eq!(bar, 7)
    }
    let u = Foo { bar: 8 };
    match Some(u) {
        Some(Foo { bar: _ }) => 3,
        None => 4,
    };

    let u = Foo { bar: 9 };
    unsafe {
        match u {
            Foo { baz: Pie { .. } } => {}
        };
    }
    let u = Foo { bar: 10 };
    unsafe {
        match u {
            Foo { baz: Pie { slices: _, size: _ } } => {}
        };
    }

    let u = Foo { bar: 11 };
    match u {
        Foo { baz: _ } => {}
    };
}
