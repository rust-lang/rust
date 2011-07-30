// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3

// Test case for issue #763, provided by robarnold.

use std;
import std::task;

tag request {
  quit;
  close(int, chan[bool]);
}

type ctx = chan[request];

fn request_task(c : chan[ctx]) {
    let p = port();
    c <| chan(p);
    let req;
    while (true) {
        p |> req;
        alt (req) {
            quit. {
                ret;
            }
            close(what, status) {
                log "closing now";
                log what;
                status <| true;
            }
        }
    }
}

fn new() -> ctx {
    let p = port();
    let t = spawn request_task(chan(p));
    let cx;
    p |> cx;
    ret cx;
}

fn main() {
    let cx = new();

    let p = port();
    cx <| close(4, chan(p));
    let result;
    p |> result;
    cx <| quit;
}
