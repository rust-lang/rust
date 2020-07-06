// check-pass

pub struct Fixed64(i64);

pub fn div(f: Fixed64) {
    f.0 / 0;
}

fn main() {}
