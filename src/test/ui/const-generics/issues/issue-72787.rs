#![feature(const_generics)]
#![allow(incomplete_features)]

pub struct IsLessOrEqual<const LHS: u32, const RHS: u32>;
pub struct Condition<const CONDITION: bool>;
pub trait True {}

impl<const LHS: u32, const RHS: u32> True for IsLessOrEqual<LHS, RHS> where
    Condition<{ LHS <= RHS }>: True
//~^ Error constant expression depends on a generic parameter
{
}
impl True for Condition<true> {}

struct S<const I: u32, const J: u32>;
impl<const I: u32, const J: u32> S<I, J>
where
    IsLessOrEqual<I, 8>: True,
    IsLessOrEqual<J, 8>: True,
    IsLessOrEqual<{ 8 - I }, { 8 - J }>: True,
//~^ Error constant expression depends on a generic parameter
//~| Error constant expression depends on a generic parameter
//~| Error constant expression depends on a generic parameter
//~| Error constant expression depends on a generic parameter
    // Condition<{ 8 - I <= 8 - J }>: True,
{
    fn print() {
        println!("I {} J {}", I, J);
    }
}

fn main() {}
