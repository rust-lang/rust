fn a() {
    let mut vec = [~1, ~2, ~3];
    match vec {
        [~ref _a] => {
            vec[0] = ~4; //~ ERROR prohibited due to outstanding loan
        }
        _ => die!(~"foo")
    }
}

fn b() {
    let mut vec = [~1, ~2, ~3];
    match vec {
        [.._b] => {
            vec[0] = ~4; //~ ERROR prohibited due to outstanding loan
        }
    }
}

fn main() {}

