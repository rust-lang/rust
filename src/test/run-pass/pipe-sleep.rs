
extern mod std;
use std::timer::sleep;
use std::uv;
use pipes::recv;

proto! oneshot (
    waiting:send {
        signal -> !
    }
)

fn main() {
    use oneshot::client::*;

    let c = pipes::spawn_service(oneshot::init, |p| { recv(move p); });

    let iotask = uv::global_loop::get();
    sleep(iotask, 500);
    
    signal(move c);
}