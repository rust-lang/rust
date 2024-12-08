pub trait ToPrimitive {
    fn to_int(&self) -> isize { 0 }
}

impl ToPrimitive for i32 {}
impl ToPrimitive for isize {}

trait Add {
    fn to_int(&self) -> isize;
    fn add_dynamic(&self, other: &dyn Add) -> isize;
}

impl Add for isize {
    fn to_int(&self) -> isize { *self }
    fn add_dynamic(&self, other: &dyn Add) -> isize {
        self.to_int() + other.to_int() //~ ERROR multiple applicable items in scope
    }
}

fn main() { }
