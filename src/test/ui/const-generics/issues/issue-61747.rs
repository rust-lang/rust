#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

struct Const<const N: usize>;

impl<const C: usize> Const<{C}> {
    fn successor() -> Const<{C + 1}> {
        //~^ ERROR constant expression depends on a generic parameter
        Const
    }
}

fn main() {
    let _x: Const::<2> = Const::<1>::successor();
}
