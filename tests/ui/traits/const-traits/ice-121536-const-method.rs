#![feature(const_trait_impl)]

pub struct Vec3;

#[const_trait]
pub trait Add {
    fn add(self) -> Vec3;
}

impl Add for Vec3 {
    const fn add(self) -> Vec3 {
        //~^ ERROR functions in trait impls cannot be declared const
        self
    }
}

fn main() {}
