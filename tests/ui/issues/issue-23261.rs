//@ run-pass
// Matching on a DST struct should not trigger an LLVM assertion.

struct Foo<T: ?Sized> {
    a: i32,
    inner: T
}

trait Get {
    fn get(&self) -> i32;
}

impl Get for i32 {
    fn get(&self) -> i32 {
        *self
    }
}

fn check_val(val: &Foo<[u8]>) {
    match *val {
        Foo { a, .. } => {
            assert_eq!(a, 32);
        }
    }
}

fn check_dst_val(val: &Foo<[u8]>) {
    match *val {
        Foo { ref inner, .. } => {
            assert_eq!(inner, [1, 2, 3]);
        }
    }
}

fn check_both(val: &Foo<[u8]>) {
    match *val {
        Foo { a, ref inner } => {
            assert_eq!(a, 32);
            assert_eq!(inner, [1, 2, 3]);
        }
    }
}

fn check_trait_obj(val: &Foo<dyn Get>) {
    match *val {
        Foo { a, ref inner } => {
            assert_eq!(a, 32);
            assert_eq!(inner.get(), 32);
        }
    }
}

fn main() {
    let foo: &Foo<[u8]> = &Foo { a: 32, inner: [1, 2, 3] };
    check_val(foo);
    check_dst_val(foo);
    check_both(foo);

    let foo: &Foo<dyn Get> = &Foo { a: 32, inner: 32 };
    check_trait_obj(foo);
}
