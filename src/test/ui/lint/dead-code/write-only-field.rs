#![deny(dead_code)]

struct S {
    f: i32, //~ ERROR: field is never read
    sub: Sub, //~ ERROR: field is never read
}

struct Sub {
    f: i32, //~ ERROR: field is never read
}

fn field_write(s: &mut S) {
    s.f = 1;
    s.sub.f = 2;
}

fn main() {
    let mut s = S { f: 0, sub: Sub { f: 0 } };
    field_write(&mut s);

    auto_deref();
    nested_boxes();
}

fn auto_deref() {
    struct E {
        x: bool,
        y: bool, //~ ERROR: field is never read
    }

    struct P<'a> {
        e: &'a mut E
    }

    impl P<'_> {
        fn f(&mut self) {
            self.e.x = true;
            self.e.y = true;
        }
    }

    let mut e = E { x: false, y: false };
    let mut p = P { e: &mut e };
    p.f();
    assert!(e.x);
}

fn nested_boxes() {
    struct A {
        b: Box<B>,
    }

    struct B {
        c: Box<C>,
    }

    struct C {
        u: u32, //~ ERROR: field is never read
        v: u32, //~ ERROR: field is never read
    }

    let mut a = A {
        b: Box::new(B {
            c: Box::new(C { u: 0, v: 0 }),
        }),
    };
    a.b.c.v = 10;
    a.b.c = Box::new(C { u: 1, v: 2 });
}
