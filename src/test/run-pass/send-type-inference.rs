// xfail-stage0
// tests that ctrl's type gets inferred properly
type command[K, V] = rec(K key, V val);

fn cache_server[K, V] (chan[chan[command[K,V]]] c) {
    auto ctrl = port();
    c <| chan(ctrl);
}
fn main() {
}