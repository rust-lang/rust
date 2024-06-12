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
    let x = match true {
        true => Default::default(),
        //[e2024]~^ error: the trait bound `!: Default` is not satisfied
        false => panic!("..."),
    };

    dbg!(x);
}

fn q() -> Option<()> {
    fn deserialize<T: Default>() -> Option<T> {
        Some(T::default())
    }

    deserialize()?;
    //[e2024]~^ error: the trait bound `!: Default` is not satisfied

    None
}
