fn main() {
    const PAT: u8 = 0;

    match 0 {
        (.. PAT) => {}
        //~^ ERROR `..X` range patterns are not supported
    }
}

const RECOVERY_WITNESS: () = 0; //~ ERROR mismatched types
