// run-pass
// check-run-results
// revisions: twenty_eighteen twenty_twentyone
// [twenty_eighteen]compile-flags: --edition 2018
// [twenty_twentyone]compile-flags: --edition 2021

#[derive(Debug)]
struct Dropable(String);

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

fn main() {
    let a = A { x: Dropable(format!("x")), y: Dropable(format!("y")) };

    let c = move || println!("{:?} {:?}", a.y, a.x);

    c();
}
