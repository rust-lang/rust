// xfail-stage0
// tests that ctrl's type gets inferred properly
type command[K, V] = {key: K, val: V};

fn cache_server[K, V](c: chan[chan[command[K, V]]]) {
    let ctrl = port();
    c <| chan(ctrl);
}
fn main() { }