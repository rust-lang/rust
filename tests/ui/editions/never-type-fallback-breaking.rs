//@ revisions: e2021 e2024
//
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//@[e2024] compile-flags: -Zunstable-options
//
//@[e2021] run-pass
//@[e2024] check-fail

fn main() {
    m();
    q();
}

fn m() {
    //[e2021]~^ this function depends on never type fallback being `()`
    //[e2021]~| this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    let x = match true {
        true => Default::default(),
        //[e2024]~^ error: the trait bound `!: Default` is not satisfied
        false => panic!("..."),
    };

    dbg!(x);
}

fn q() -> Option<()> {
    //[e2021]~^ this function depends on never type fallback being `()`
    //[e2021]~| this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    fn deserialize<T: Default>() -> Option<T> {
        Some(T::default())
    }

    deserialize()?;
    //[e2024]~^ error: the trait bound `!: Default` is not satisfied

    None
}
