use std;
import std::comm::chan;
import std::comm::send;
import std::comm::port;

// tests that ctrl's type gets inferred properly
type command<send K, send V> = {key: K, val: V};

fn cache_server<send K, send V>(c: chan<chan<command<K, V>>>) {
    let ctrl = port();
    send(c, chan(ctrl));
}
fn main() { }
