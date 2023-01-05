// run-pass
pub trait NumCast: Sized {
    fn from(i: i32) -> Option<Self>;
}

pub trait NumExt: PartialEq + PartialOrd + NumCast {}

impl NumExt for f32 {}
impl NumCast for f32 {
    fn from(i: i32) -> Option<f32> { Some(i as f32) }
}

fn num_eq_one<T: NumExt>(n: T) {
    println!("{}", n == NumCast::from(1).unwrap())
}

pub fn main() {
    num_eq_one(1f32); // you need to actually use the function to trigger the ICE
}
