use task::*;
use comm::*;

struct test {
  let f: int;
  new(f: int) { self.f = f; }
  drop {}
}

fn main() {
    let p = Port();
    let c = Chan(p);

    do spawn() {
        let p = Port();
        c.send(Chan(p));

        let _r = p.recv();
    }

    p.recv().send(test(42));
}