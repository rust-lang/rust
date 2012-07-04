import task::*;
import comm::*;

class test {
  let f: int;
  new(f: int) { self.f = f; }
  drop {}
}

fn main() {
    let p = port();
    let c = chan(p);

    do spawn() {
        let p = port();
        c.send(chan(p));

        let _r = p.recv();
    }

    p.recv().send(test(42));
}