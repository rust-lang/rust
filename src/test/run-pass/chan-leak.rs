// Issue #763

use std;
use comm::Chan;
use comm::send;
use comm::Port;
use comm::recv;

enum request { quit, close(Chan<bool>), }

type ctx = Chan<request>;

fn request_task(c: Chan<ctx>) {
    let p = Port();
    send(c, Chan(p));
    let mut req: request;
    req = recv(p);
    // Need to drop req before receiving it again
    req = recv(p);
}

fn new_cx() -> ctx {
    let p = Port();
    let ch = Chan(p);
    let t = task::spawn(|| request_task(ch) );
    let mut cx: ctx;
    cx = recv(p);
    return cx;
}

fn main() {
    let cx = new_cx();

    let p = Port::<bool>();
    send(cx, close(Chan(p)));
    send(cx, quit);
}
