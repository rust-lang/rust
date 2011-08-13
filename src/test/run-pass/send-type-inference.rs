use std;
import std::comm::_chan;
import std::comm::send;
import std::comm::mk_port;

// tests that ctrl's type gets inferred properly
type command<K, V> = {key: K, val: V};

fn cache_server<K, V>(c: _chan<_chan<command<K, V>>>) {
    let ctrl = mk_port::<_chan<command<K, V>>>();
    send(c, ctrl.mk_chan());
}
fn main() { }
