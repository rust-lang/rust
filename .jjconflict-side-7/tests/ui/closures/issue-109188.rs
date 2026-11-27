enum Either {
    One(X),
    Two(X),
}

struct X(Y);

struct Y;

fn consume_fnmut(f: &mut dyn FnMut()) {
    f();
}

fn move_into_fnmut() {
    let x = move_into_fnmut();
    consume_fnmut(&mut || {
        let Either::One(_t) = x; //~ ERROR mismatched types
        let Either::Two(_t) = x; //~ ERROR mismatched types
    });
}

fn main() { }
