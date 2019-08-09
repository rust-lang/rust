// build-pass (FIXME(62277): could be check-pass?)

use std::collections::HashMap;
use std::sync::Mutex;

fn i_used_to_be_able_to(foo: &Mutex<HashMap<usize, usize>>) -> Vec<(usize, usize)> {
    let mut foo = foo.lock().unwrap();

    foo.drain().collect()
}

fn but_after_nightly_update_now_i_gotta(foo: &Mutex<HashMap<usize, usize>>) -> Vec<(usize, usize)> {
    let mut foo = foo.lock().unwrap();

    return foo.drain().collect();
}

fn main() {}
