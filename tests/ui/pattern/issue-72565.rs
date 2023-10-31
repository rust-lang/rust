const F: &'static dyn PartialEq<u32> = &7u32;

fn main() {
    let a: &dyn PartialEq<u32> = &7u32;
    match a {
        F => panic!(), //~ ERROR: `dyn PartialEq<u32>` cannot be used in patterns
    }
}
