// Issue #763

use std;
import std::task;

tag request {
  quit;
  close(chan[bool]);
}

type ctx = chan[request];

fn request_task(c: chan[ctx]) {
    let p: port[request] = port();
    c <| chan(p);
    let req: request;
    p |> req;
    // Need to drop req before receiving it again
    p |> req;
}

fn new() -> ctx {
    let p: port[ctx] = port();
    let t = spawn request_task(chan(p));
    let cx: ctx;
    p |> cx;
    ret cx;
}

fn main() {
    let cx = new();

    let p: port[bool] = port();
    cx <| close(chan(p));
    cx <| quit;
}
