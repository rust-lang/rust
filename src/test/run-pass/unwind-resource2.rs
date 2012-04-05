// xfail-win32
use std;
import task;
import comm;

resource complainer(c: @int) {
}

fn f() {
    let c <- complainer(@0);
    fail;
}

fn main() {
    let builder = task::builder();
    task::unsupervise(builder);
    task::run(builder) {|| f(); }
}