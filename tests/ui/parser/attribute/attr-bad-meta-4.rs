macro_rules! mac {
    ($attr_item: meta) => {
        #[cfg($attr_item)]
        //~^ ERROR expected unsuffixed literal, found `an(arbitrary token stream)`
        //~| ERROR expected unsuffixed literal, found `an(arbitrary token stream)`
        struct S;
    }
}

mac!(an(arbitrary token stream));

#[cfg(feature = -1)]
//~^ ERROR expected unsuffixed literal, found `-`
//~| ERROR expected unsuffixed literal, found `-`
fn handler() {}

fn main() {}
