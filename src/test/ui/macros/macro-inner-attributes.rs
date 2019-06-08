#![feature(rustc_attrs)]

macro_rules! test { ($nm:ident,
                     #[$a:meta],
                     $i:item) => (mod $nm { #![$a] $i }); }

test!(a,
      #[cfg(qux)],
      pub fn bar() { });

test!(b,
      #[cfg(not(qux))],
      pub fn bar() { });

#[rustc_dummy]
fn main() {
    a::bar();
    //~^ ERROR failed to resolve: use of undeclared type or module `a`
    b::bar();
}
