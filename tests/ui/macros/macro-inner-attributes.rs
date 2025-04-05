#![feature(rustc_attrs)]

macro_rules! test { ($nm:ident,
                     #[$a:meta],
                     $i:item) => (mod $nm { #![$a] $i }); }

test!(a,
      #[cfg(false)],
      pub fn bar() { });

test!(b,
      #[cfg(not(FALSE))],
      pub fn bar() { });

#[rustc_dummy]
fn main() {
    a::bar();
    //~^ ERROR failed to resolve: use of unresolved module or unlinked crate `a`
    b::bar();
}
