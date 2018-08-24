use std::ops::AddAssign;

struct Int(i32);

impl AddAssign<i32> for Int {
    fn add_assign(&mut self, _: i32) {
    }
}

fn main() {
    let mut x = Int(0);
    x += 1;
}
