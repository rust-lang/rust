fn main() {
    const PAT: u8 = 0;

    match 0 {
        (.. PAT) => {}
        //~^ ERROR `..X` range patterns are not supported
        //~| ERROR exclusive range pattern syntax is experimental
    }
}

const RECOVERY_WITNESS: () = 0; //~ ERROR mismatched types
