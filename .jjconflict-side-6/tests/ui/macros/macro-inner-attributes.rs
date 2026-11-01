#![feature(rustc_attrs)]

macro_rules! test { ($nm:ident,
                     #[$a:meta],
                     $i:item) => (mod $nm { #![$a] $i }); }

test!(a, //~ NOTE: found an item that was configured out
      #[cfg(false)], //~ NOTE: the item is gated here
      pub fn bar() { });

test!(b,
      #[cfg(not(FALSE))],
      pub fn bar() { });

#[rustc_dummy]
fn main() {
    a::bar(); //~ ERROR: cannot find module or crate `a`
    //~^ NOTE: use of unresolved module or unlinked crate `a`
    b::bar();
}
