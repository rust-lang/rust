impl X {
    //~^ ERROR cannot find type
    fn getn<const N: usize>() -> [u8; N] {
        getn::<N>()
    }
}
fn getn<const N: cfg_attr>() -> [u8; N] {}
//~^ ERROR: cannot find type `cfg_attr` in this scope
//~| ERROR: mismatched types

fn main() {}
