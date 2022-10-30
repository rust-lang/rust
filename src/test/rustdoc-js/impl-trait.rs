pub trait Aaaaaaa {}

impl Aaaaaaa for () {}

pub fn bbbbbbb() -> impl Aaaaaaa {
    ()
}

pub struct Ccccccc {}

impl Ccccccc {
    pub fn ddddddd(&self) -> impl Aaaaaaa {
        ()
    }
    pub fn eeeeeee(&self, _x: impl Aaaaaaa) -> i32 {
        0
    }
    pub fn fffffff(&self, x: impl Aaaaaaa) -> impl Aaaaaaa {
        x
    }
}
