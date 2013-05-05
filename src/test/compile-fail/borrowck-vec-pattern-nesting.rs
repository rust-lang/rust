fn a() {
    let mut vec = [~1, ~2, ~3];
    match vec {
        [~ref _a] => {
            vec[0] = ~4; //~ ERROR cannot assign to `vec[]` because it is borrowed
        }
        _ => fail!(~"foo")
    }
}

fn b() {
    let mut vec = [~1, ~2, ~3];
    match vec {
        [.._b] => {
            vec[0] = ~4; //~ ERROR cannot assign to `vec[]` because it is borrowed
        }
    }
}

fn main() {}
