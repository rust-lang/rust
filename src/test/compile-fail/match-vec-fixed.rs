fn a() {
    let v = [1, 2, 3];
    match v {
        [_, _, _] => {}
        [_, _, _] => {} //~ ERROR unreachable pattern
    }
    match v {
        [_, 1, _] => {}
        [_, 1, _] => {} //~ ERROR unreachable pattern
        _ => {}
    }
}

fn main() {
    a();
}
