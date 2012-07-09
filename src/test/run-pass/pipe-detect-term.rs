// Make sure that we can detect when one end of the pipe is closed.

// xfail-pretty

use std;
import std::timer::sleep;
import std::uv;

import pipes::{recv};

proto! oneshot {
    waiting:send {
        signal -> signaled
    }

    signaled:send { }
}

fn main() {
    let iotask = uv::global_loop::get();
    
    let c = pipes::spawn_service(oneshot::init, |p| { 
        alt recv(p) {
          some(*) { fail }
          none { }
        }
    });

    sleep(iotask, 1000);
}
