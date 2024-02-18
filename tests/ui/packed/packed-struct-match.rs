//@ run-pass

#[repr(packed)]
struct Foo1 {
    bar: u8,
    baz: usize
}

#[repr(packed(2))]
struct Foo2 {
    bar: u8,
    baz: usize
}

#[repr(C, packed(4))]
struct Foo4C {
    bar: u8,
    baz: usize
}

pub fn main() {
    let foo1 = Foo1 { bar: 1, baz: 2 };
    match foo1 {
        Foo1 {bar, baz} => {
            assert_eq!(bar, 1);
            assert_eq!(baz, 2);
        }
    }

    let foo2 = Foo2 { bar: 1, baz: 2 };
    match foo2 {
        Foo2 {bar, baz} => {
            assert_eq!(bar, 1);
            assert_eq!(baz, 2);
        }
    }

    let foo4 = Foo4C { bar: 1, baz: 2 };
    match foo4 {
        Foo4C {bar, baz} => {
            assert_eq!(bar, 1);
            assert_eq!(baz, 2);
        }
    }
}
