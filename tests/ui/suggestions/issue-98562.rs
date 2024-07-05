//@ aux-build:extern-issue-98562.rs

extern crate extern_issue_98562;
use extern_issue_98562::TraitA;

struct X;
impl TraitA<u8, u16, u32> for X {
    //~^ ERROR not all trait items implemented
}
//~^ HELP implement the missing item

fn main() {}
