//@ [full] check-pass
//@ revisions: full min
#![cfg_attr(full, feature(generic_const_exprs))]
#![cfg_attr(full, allow(incomplete_features))]

pub struct IsLessOrEqual<const LHS: u32, const RHS: u32>;
pub struct Condition<const CONDITION: bool>;
pub trait True {}

impl<const LHS: u32, const RHS: u32> True for IsLessOrEqual<LHS, RHS> where
    Condition<{ LHS <= RHS }>: True
//[min]~^ ERROR generic parameters may not be used in const operations
//[min]~| ERROR generic parameters may not be used in const operations
{
}
impl True for Condition<true> {}

struct S<const I: u32, const J: u32>;
impl<const I: u32, const J: u32> S<I, J>
where
    IsLessOrEqual<I, 8>: True,
    IsLessOrEqual<J, 8>: True,
    IsLessOrEqual<{ 8 - I }, { 8 - J }>: True,
//[min]~^ ERROR generic parameters may not be used in const operations
//[min]~| ERROR generic parameters may not be used in const operations
    // Condition<{ 8 - I <= 8 - J }>: True,
{
    fn print() {
        println!("I {} J {}", I, J);
    }
}

fn main() {}
