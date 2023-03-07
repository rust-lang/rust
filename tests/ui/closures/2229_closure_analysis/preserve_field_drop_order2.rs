// run-pass
// check-run-results
// revisions: twenty_eighteen twenty_twentyone
// [twenty_eighteen]compile-flags: --edition 2018
// [twenty_twentyone]compile-flags: --edition 2021

#[derive(Debug)]
struct Dropable(&'static str);

impl Drop for Dropable {
    fn drop(&mut self) {
        println!("Dropping {}", self.0)
    }
}

#[derive(Debug)]
struct A {
    x: Dropable,
    y: Dropable,
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
    let a = A { x: Dropable("x"), y: Dropable("y") };

    let c = move || println!("{:?} {:?}", a.y, a.x);

    c();

    let b = B {
        c: A { x: Dropable("b.c.x"), y: Dropable("b.c.y") },
        d: A { x: Dropable("b.d.x"), y: Dropable("b.d.y") },
    };

    let d = move || println!("{:?} {:?} {:?} {:?}", b.d.y, b.d.x, b.c.y, b.c.x);

    d();

        let r = R {
        c: &A { x: Dropable("r.c.x"), y: Dropable("r.c.y") },
        d: &A { x: Dropable("r.d.x"), y: Dropable("r.d.y") },
    };

    let e = move || println!("{:?} {:?} {:?} {:?}", r.d.y, r.d.x, r.c.y, r.c.x);

    e();
}
