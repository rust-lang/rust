use std::ops::AddAssign;

pub struct Int(pub i32);

impl AddAssign<i32> for Int {
    fn add_assign(&mut self, _: i32) {
    }
}
