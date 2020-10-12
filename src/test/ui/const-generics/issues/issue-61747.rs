// revisions: full min
#![cfg_attr(full, feature(const_generics))] //[full]~WARN the feature `const_generics` is incomplete
#![cfg_attr(min, feature(min_const_generics))]

struct Const<const N: usize>;

impl<const C: usize> Const<{C}> {
    fn successor() -> Const<{C + 1}> {
        //[full]~^ ERROR constant expression depends on a generic parameter
        //[min]~^^ ERROR generic parameters may not be used
        Const
    }
}

fn main() {
    let _x: Const::<2> = Const::<1>::successor();
}
