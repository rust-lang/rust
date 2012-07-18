// xfail-pretty

use std;
import std::timer::sleep;
import std::uv;

proto! oneshot {
    waiting:send {
        signal -> !
    }
}

fn main() {
    let (c, p) = oneshot::init();

    assert !pipes::peek(p);

    oneshot::client::signal(c);

    assert pipes::peek(p);
}
