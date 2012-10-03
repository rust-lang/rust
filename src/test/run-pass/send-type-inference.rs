extern mod std;
use comm::Chan;
use comm::send;
use comm::Port;

// tests that ctrl's type gets inferred properly
type command<K: Send, V: Send> = {key: K, val: V};

fn cache_server<K: Send, V: Send>(c: Chan<Chan<command<K, V>>>) {
    let ctrl = Port();
    send(c, Chan(&ctrl));
}
fn main() { }
