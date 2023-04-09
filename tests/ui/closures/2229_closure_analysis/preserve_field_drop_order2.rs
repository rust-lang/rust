// run-pass
// check-run-results
// revisions: twenty_eighteen twenty_twentyone
// [twenty_eighteen]compile-flags: --edition 2018
// [twenty_twentyone]compile-flags: --edition 2021

#[derive(Debug)]
struct Droppable(&'static str);

impl Drop for Droppable {
    fn drop(&mut self) {
        println!("Dropping {}", self.0)
    }
}

#[derive(Debug)]
struct A {
    x: Droppable,
    y: Droppable,
}

#[derive(Debug)]
struct B {
    c: A,
    d: A,
}

#[derive(Debug)]
struct R<'a> {
    c: &'a A,
    d: &'a A,
}

fn main() {
    let a = A { x: Droppable("x"), y: Droppable("y") };

    let c = move || println!("{:?} {:?}", a.y, a.x);

    c();

    let b = B {
        c: A { x: Droppable("b.c.x"), y: Droppable("b.c.y") },
        d: A { x: Droppable("b.d.x"), y: Droppable("b.d.y") },
    };

    let d = move || println!("{:?} {:?} {:?} {:?}", b.d.y, b.d.x, b.c.y, b.c.x);

    d();

        let r = R {
        c: &A { x: Droppable("r.c.x"), y: Droppable("r.c.y") },
        d: &A { x: Droppable("r.d.x"), y: Droppable("r.d.y") },
    };

    let e = move || println!("{:?} {:?} {:?} {:?}", r.d.y, r.d.x, r.c.y, r.c.x);

    e();
}
