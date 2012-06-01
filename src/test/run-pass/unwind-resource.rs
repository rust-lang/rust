// xfail-win32
use std;
import task;
import comm;

class complainer {
  let c: comm::chan<bool>;
  new(c: comm::chan<bool>) {
    #error("Hello!");
    self.c = c; }
  drop { #error("About to send!");
    comm::send(self.c, true);
    #error("Sent!"); }
}

fn f(c: comm::chan<bool>) {
    let _c <- complainer(c);
    fail;
}

fn main() {
    let p = comm::port();
    let c = comm::chan(p);
    let builder = task::builder();
    task::unsupervise(builder);
    task::run(builder) {|| f(c); }
    #error("hiiiiiiiii");
    assert comm::recv(p);
}