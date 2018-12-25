// run-pass
pub struct _X([u8]);

impl std::ops::Deref for _X {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.0
    }
}

pub fn _g(x: &_X) -> &[u8] {
    x
}

fn main() {
}
