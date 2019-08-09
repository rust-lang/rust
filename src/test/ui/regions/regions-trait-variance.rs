#![feature(box_syntax)]

// Issue #12470.

trait X {
    fn get_i(&self) -> isize;
}

struct B {
    i: isize
}

impl X for B {
    fn get_i(&self) -> isize {
        self.i
    }
}

impl Drop for B {
    fn drop(&mut self) {
        println!("drop");
    }
}

struct A<'r> {
    p: &'r (dyn X + 'r)
}

fn make_a(p: &dyn X) -> A {
    A{p:p}
}

fn make_make_a<'a>() -> A<'a> {
    let b: Box<B> = box B {
        i: 1,
    };
    let bb: &B = &*b;
    make_a(bb) //~ ERROR cannot return value referencing local data `*b`
}

fn main() {
    let a = make_make_a();
    println!("{}", a.p.get_i());
}
