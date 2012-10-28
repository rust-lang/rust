// Make sure that we can detect when one end of the pipe is closed.

// xfail-win32

extern mod std;
use std::timer::sleep;
use std::uv;

use pipes::{try_recv, recv};

proto! oneshot (
    waiting:send {
        signal -> !
    }
)

fn main() {
    let iotask = uv::global_loop::get();
    
    pipes::spawn_service(oneshot::init, |p| { 
        match try_recv(move p) {
          Some(*) => { fail }
          None => { }
        }
    });

    sleep(iotask, 100);

    task::spawn_unlinked(failtest);
}

// Make sure the right thing happens during failure.
fn failtest() {
    let (c, p) = oneshot::init();

    do task::spawn_with(move c) |_c| { 
        fail;
    }

    error!("%?", recv(move p));
    // make sure we get killed if we missed it in the receive.
    loop { task::yield() }
}
