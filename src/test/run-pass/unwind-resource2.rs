// xfail-win32
use std;
import task;
import comm;

class complainer {
  let c: @int;
  new(c: @int) { self.c = c; }
  drop {}
}

fn f() {
    let c <- complainer(@0);
    fail;
}

fn main() {
    let builder = task::builder();
    task::unsupervise(builder);
    task::run(builder, || f() );
}