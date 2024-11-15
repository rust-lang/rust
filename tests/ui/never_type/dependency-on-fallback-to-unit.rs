//@ check-pass

fn main() {
    def();
    _ = question_mark();
}

fn def() {
    //~^ warn: this function depends on never type fallback being `()`
    //~| warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in Rust 2024 and in a future release in all editions!
    match true {
        false => <_>::default(),
        true => return,
    };
}

// <https://github.com/rust-lang/rust/issues/51125>
// <https://github.com/rust-lang/rust/issues/39216>
fn question_mark() -> Result<(), ()> {
    //~^ warn: this function depends on never type fallback being `()`
    //~| warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in Rust 2024 and in a future release in all editions!
    deserialize()?;
    Ok(())
}

fn deserialize<T: Default>() -> Result<T, ()> {
    Ok(T::default())
}
