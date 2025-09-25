// Regression test for https://github.com/rust-lang/rust/issues/138510

#![feature(where_clause_attrs)]
#![deny(unused_attributes)]

fn main() {
}

fn test() where
#[repr()]
//~^ ERROR unused attribute
(): Sized {

}
