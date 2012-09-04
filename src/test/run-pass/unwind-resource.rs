// xfail-win32
use std;
import task;
import comm;

struct complainer {
  let c: comm::Chan<bool>;
  new(c: comm::Chan<bool>) {
    error!("Hello!");
    self.c = c; }
  drop { error!("About to send!");
    comm::send(self.c, true);
    error!("Sent!"); }
}

fn f(c: comm::Chan<bool>) {
    let _c <- complainer(c);
    fail;
}

fn main() {
    let p = comm::Port();
    let c = comm::Chan(p);
    task::spawn_unlinked(|| f(c) );
    error!("hiiiiiiiii");
    assert comm::recv(p);
}
