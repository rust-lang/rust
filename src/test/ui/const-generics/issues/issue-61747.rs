// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct Const<const N: usize>;

impl<const C: usize> Const<{C}> {
    fn successor() -> Const<{C + 1}> {
        Const
    }
}

fn main() {
    Const::<1>::successor();
}
