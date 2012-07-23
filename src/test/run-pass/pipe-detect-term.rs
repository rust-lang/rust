// Make sure that we can detect when one end of the pipe is closed.

// xfail-pretty
// xfail-win32

use std;
import std::timer::sleep;
import std::uv;

import pipes::{try_recv, recv};

proto! oneshot {
    waiting:send {
        signal -> !
    }
}

fn main() {
    let iotask = uv::global_loop::get();
    
    pipes::spawn_service(oneshot::init, |p| { 
        alt try_recv(p) {
          some(*) { fail }
          none { }
        }
    });

    sleep(iotask, 100);

    task::spawn_unlinked(failtest);
}

// Make sure the right thing happens during failure.
fn failtest() {
    let (c, p) = oneshot::init();

    do task::spawn_with(c) |_c| { 
        fail;
    }

    #error("%?", recv(p));
    // make sure we get killed if we missed it in the receive.
    loop { task::yield() }
}
