macro_rules! mac {
    ($attr_item: meta) => {
        #[cfg($attr_item)]
        //~^ ERROR expected unsuffixed literal or identifier, found `an(arbitrary token stream)`
        //~| ERROR expected unsuffixed literal or identifier, found `an(arbitrary token stream)`
        struct S;
    }
}

mac!(an(arbitrary token stream));

fn main() {}
