// xfail-win32
use std;

struct complainer {
  c: comm::Chan<bool>,
  drop { error!("About to send!");
    comm::send(self.c, true);
    error!("Sent!"); }
}

fn complainer(c: comm::Chan<bool>) -> complainer {
    error!("Hello!");
    complainer {
        c: c
    }
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
