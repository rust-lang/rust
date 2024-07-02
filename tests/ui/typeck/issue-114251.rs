// compile-flags: -C debug-assertions

macro_rules! feedu8 {
    () => {
        0u8
    }
}

fn main() {
    let kitty = {
        feedu8!();
        //~^ HELP: remove this semicolon
    };

    let cat: u8 = kitty;
    //~^ ERROR: mismatched types
}
