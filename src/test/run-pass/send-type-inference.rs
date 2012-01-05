use std;
import comm::chan;
import comm::send;
import comm::port;

// tests that ctrl's type gets inferred properly
type command<K: send, V: send> = {key: K, val: V};

fn cache_server<K: send, V: send>(c: chan<chan<command<K, V>>>) {
    let ctrl = port();
    send(c, chan(ctrl));
}
fn main() { }
