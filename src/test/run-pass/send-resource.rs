use task::*;
use comm::*;

struct test {
  let f: int;
  drop {}
}

fn test(f: int) -> test {
    test {
        f: f
    }
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