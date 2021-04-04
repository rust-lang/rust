// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

pub struct IsLessOrEqual<const LHS: u32, const RHS: u32>;
pub struct Condition<const CONDITION: bool>;
pub trait True {}

impl<const LHS: u32, const RHS: u32> True for IsLessOrEqual<LHS, RHS> where
    Condition<{ LHS <= RHS }>: True
//[full]~^ Error constant expression depends on a generic parameter
//[min]~^^ Error generic parameters may not be used in const operations
//[min]~| Error generic parameters may not be used in const operations
{
}
impl True for Condition<true> {}

struct S<const I: u32, const J: u32>;
impl<const I: u32, const J: u32> S<I, J>
where
    IsLessOrEqual<I, 8>: True,
//[min]~^ Error type annotations needed [E0283]
//[min]~| Error type annotations needed [E0283]
    IsLessOrEqual<J, 8>: True,
    IsLessOrEqual<{ 8 - I }, { 8 - J }>: True,
//[full]~^ constant expression depends on a generic parameter
//[full]~| constant expression depends on a generic parameter
//[full]~| constant expression depends on a generic parameter
//[full]~| constant expression depends on a generic parameter
//[min]~^^^^^ Error generic parameters may not be used in const operations
//[min]~| Error generic parameters may not be used in const operations
    // Condition<{ 8 - I <= 8 - J }>: True,
{
    fn print() {
        println!("I {} J {}", I, J);
    }
}

fn main() {}
