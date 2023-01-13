// run-pass

struct A { a: Box<i32> }

impl A {
    fn get(&self) -> i32 { *self.a }
}

fn foo(n: i32) -> i32 {
    let x = A { a: Box::new(n) };
    let y = match x {
        A { a: v } if *v == 42 => v,
        _ => Box::new(0),
    };
    *y
}

fn bar(n: i32) -> i32 {
    let x = A { a: Box::new(n) };
    let y = match x {
        A { a: v } if x.get() == 42 => v,
        _ => Box::new(0),
    };
    *y
}

fn baz(n: i32) -> i32 {
    let x = A { a: Box::new(n) };
    let y = match x {
        A { a: v } if *v.clone() == 42 => v,
        _ => Box::new(0),
    };
    *y
}

fn main() {
    assert_eq!(foo(107), 0);
    assert_eq!(foo(42), 42);
    assert_eq!(bar(107), 0);
    assert_eq!(bar(42), 42);
    assert_eq!(baz(107), 0);
    assert_eq!(baz(42), 42);
}
