// run-pass

fn vec() {
    let mut _x = vec!['c'];
    let _y = &_x;
    _x = Vec::new();
}

fn int() {
    let mut _x = 5;
    let _y = &_x;
    _x = 7;
}

fn main() {
    vec();
    int();
}
