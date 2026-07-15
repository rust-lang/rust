//@ run-rustfix

macro_rules! foo {
    () => {
        assert_eq!("A", "A");
        //~^ ERROR macro expansion ignores `;`
        //~| NOTE the usage of `foo!` is likely invalid in expression context
        assert_eq!("B", "B");
    }
}

fn main() {
    foo!()
    //~^ NOTE caused by the macro expansion here
}
