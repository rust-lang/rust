//@ aux-build:extern-issue-98562.rs

extern crate extern_issue_98562;
use extern_issue_98562::TraitA;

struct X;
impl TraitA<u8, u16, u32> for X {
    //~^ ERROR not all trait items implemented
}
//~^ HELP implement the missing item: `fn baz<U: TraitC<I1 = u8, I2 = u16> + TraitD<I3 = u32>, V: TraitD<I3 = u8>>(_: U, _: V) -> Self where U: TraitE, U: TraitB, <U as TraitB>::Item: Copy { todo!() }`

fn main() {}
