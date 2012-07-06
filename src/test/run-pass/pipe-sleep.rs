// xfail-pretty

use std;
import std::timer::sleep;
import std::uv;
import pipes::recv;

proto! oneshot {
    waiting:send {
        signal -> signaled
    }

    signaled:send { }
}

fn main() {
    import oneshot::client::*;

    let c = pipes::spawn_service(oneshot::init, |p| { recv(p); });

    let iotask = uv::global_loop::get();
    sleep(iotask, 5000);
    
    signal(c);
}