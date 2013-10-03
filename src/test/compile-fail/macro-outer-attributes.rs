#[feature(macro_rules)];

macro_rules! test ( ($nm:ident,
                     $a:attr,
                     $i:item) => (mod $nm { $a $i }); )

test!(a,
      #[cfg(qux)],
      pub fn bar() { })

test!(b,
      #[cfg(not(qux))],
      pub fn bar() { })

// test1!(#[bar])
#[qux]
fn main() {
    a::bar(); //~ ERROR unresolved name `a::bar`
    b::bar();
}

