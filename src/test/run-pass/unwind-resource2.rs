// xfail-win32
use std;

struct complainer {
  let c: @int;
  drop {}
}

fn complainer(c: @int) -> complainer {
    complainer {
        c: c
    }
}

fn f() {
    let c <- complainer(@0);
    fail;
}

fn main() {
    task::spawn_unlinked(f);
}
