use std;
import std::comm::chan;
import std::comm::send;
import std::comm::port;

// tests that ctrl's type gets inferred properly
type command<uniq K, uniq V> = {key: K, val: V};

fn cache_server<uniq K, uniq V>(c: chan<chan<command<K, V>>>) {
    let ctrl = port();
    send(c, chan(ctrl));
}
fn main() { }
