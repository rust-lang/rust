// This is a test for various ways in which the change to the never type
// fallback can break things.
//
// It also used to test for the `dependency_on_unit_never_type_fallback` lint.
//
//@ revisions: e2021 e2024
//
//@[e2021] edition: 2021
//@[e2024] edition: 2024

fn main() {
    m();
    q();
    let _ = meow();
    let _ = fallback_return();
    let _ = fully_apit();
}

fn m() {
    let x = match true {
        true => Default::default(),
        //~^ error: the trait bound `!: Default` is not satisfied
        false => panic!("..."),
    };

    dbg!(x);
}

fn q() -> Option<()> {
    fn deserialize<T: Default>() -> Option<T> {
        Some(T::default())
    }

    deserialize()?;
    //~^ error: the trait bound `!: Default` is not satisfied

    None
}

// Make sure we turbofish the right argument
fn help<'a: 'a, T: Into<()>, U>(_: U) -> Result<T, ()> {
    Err(())
}
fn meow() -> Result<(), ()> {
    help(1)?;
    //~^ error: the trait bound `(): From<!>` is not satisfied
    Ok(())
}

pub fn takes_apit<T>(_y: impl Fn() -> T) -> Result<T, ()> {
    Err(())
}

pub fn fallback_return() -> Result<(), ()> {
    takes_apit(|| Default::default())?;
    //~^ error: the trait bound `!: Default` is not satisfied
    Ok(())
}

fn mk<T>() -> Result<T, ()> {
    Err(())
}

fn takes_apit2(_x: impl Default) {}

fn fully_apit() -> Result<(), ()> {
    takes_apit2(mk()?);
    //~^ error: the trait bound `!: Default` is not satisfied
    Ok(())
}
