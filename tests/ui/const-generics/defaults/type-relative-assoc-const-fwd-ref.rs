// Ensure that we detect and reject implicit forward references to generic parameters in const
// parameter defaults arising from type-relative associated item paths.

#![feature(gca_min_const_items, gca_macroless_args)]

// Below, `T::CT` resolves to `<T as Trait<N>>::CT` but `N` is a forward reference.
struct Def0<T: Trait<N>, const N: usize = { T::CT }>(T);
//~^ ERROR const parameter default references generic parameter `N` before it is declared
struct Use0(Def0<()>);

trait Trait<const N: usize> {
    #[rustc_always_gca]
    const CT: usize;
}

fn main() {}
