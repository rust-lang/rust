fn a() {
    let mut v = ~[1, 2, 3];
    match v {
        [_a, ..tail] => {
            v.push(tail[0] + tail[1]); //~ ERROR cannot borrow
        }
        _ => {}
    };
}

fn main() {}
