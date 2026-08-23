//! Checks that `#[rustc_trivial_field_reads]` applies per method
//! (issue #160621)

#![feature(rustc_attrs)]
#![deny(dead_code)]

trait Access {
    fn get_a(&self) -> u32;
    fn get_b(&self) -> u32;
}

struct S {
    a: u32, //~ ERROR field `a` is never read
    b: u32
}

impl Access for S {
    #[rustc_trivial_field_reads]
    fn get_a(&self) -> u32 {
        self.a
    }

    fn get_b(&self) -> u32 {
        self.b
    }
}

struct T {
    a: u32,
    b: u32
}

impl Access for T {
    fn get_a(&self) -> u32 {
        self.a
    }

    fn get_b(&self) -> u32 {
        self.b
    }
}

struct Square {
    width: u32, //~ ERROR field `width` is never read
    height: u32
}

impl Square {
    #[rustc_trivial_field_reads]
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }
}

struct U {
    a: u32, //~ ERROR field `a` is never read
    b: u32
}

#[rustc_trivial_field_reads]
fn foo(u: &U) -> u32 {
    u.a
}

fn bar(u: &U) -> u32 {
    u.b
}

fn main() {
    let s = S {
        a: 0,
        b: 0
    };

    let _ = s.get_a();
    let _ = s.get_b();

    let t = T {
        a: 0,
        b: 0
    };

    let _ = t.get_a();
    let _ = t.get_b();

    let square = Square { width: 0, height: 0 };
    let _ = square.width();
    let _ = square.height();

    let u = U { a: 0, b:0 };
    let _ = foo(&u);
    let _ = bar(&u);
}
