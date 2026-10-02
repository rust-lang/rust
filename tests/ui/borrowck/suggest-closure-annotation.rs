use std::collections::HashMap;

fn original() {
    let map = HashMap::<usize, usize>::new();
    let f = |v| map[v];
    for a in 0..1 {
        f(&a); //~ ERROR `a` does not live long enough
    }
}

fn multiple_params() {
    let map = HashMap::<usize, usize>::new();
    let f = |v, w| map[v] + w;
    for a in 0..1 {
        f(&a, 2); //~ ERROR `a` does not live long enough
    }
}

fn partial_type() {
    let map = HashMap::<usize, usize>::new();
    let f = |v: _| map[v];
    for a in 0..1 {
        f(&a); //~ ERROR `a` does not live long enough
    }
}

fn main() {
    original();
    multiple_params();
    partial_type();
}
