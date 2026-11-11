// The span of the suggestion should be correct and not ICE on this code (#161472)
macro_rules! m { ($m:meta) => { #[derive($m)] pub struct S; }; }

m!(a(::b::c));
//~^ ERROR traits in `#[derive(...)]` don't accept arguments
//~| ERROR cannot find derive macro `a` in this scope
//~| ERROR cannot find derive macro `a` in this scope
fn main(){}
