macro_rules! mac {
    ($attr_item: meta) => {
        #[cfg($attr_item)]
        //~^ ERROR expected unsuffixed literal, found `meta` metavariable
        //~| ERROR expected unsuffixed literal, found `meta` metavariable
        struct S;
    }
}

mac!(an(arbitrary token stream));

#[cfg(feature = -1)]
//~^ ERROR expected unsuffixed literal, found `-`
//~| ERROR expected unsuffixed literal, found `-`
fn handler() {}

fn main() {}
