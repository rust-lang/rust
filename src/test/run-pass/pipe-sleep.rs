// xfail-pretty

use std;
import std::timer::sleep;
import std::uv;
import pipes::recv;

proto! oneshot {
    waiting:send {
        signal -> !
    }
}

fn main() {
    import oneshot::client::*;

    let c = pipes::spawn_service(oneshot::init, |p| { recv(p); });

    let iotask = uv::global_loop::get();
    sleep(iotask, 500);
    
    signal(c);
}