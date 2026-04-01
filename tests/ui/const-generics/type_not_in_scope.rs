impl X {
    //~^ ERROR cannot find type
    fn getn<const N: usize>() -> [u8; N] {
        getn::<N>()
    }
}
fn getn<const N: cfg_attr>() -> [u8; N] {}
//~^ ERROR: expected type, found built-in attribute `cfg_attr`
//~| ERROR: mismatched types

fn main() {}
