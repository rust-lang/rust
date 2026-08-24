fn main() {
    let addr = Into::<std::net::IpAddr>.into([127, 0, 0, 1]);
    //~^ ERROR cannot find value `Into` in this scope
    //~| HELP use the path separator

    let _ = Into.into(());
    //~^ ERROR cannot find value `Into` in this scope
    //~| HELP use the path separator

    let _ = Into::<()>.into;
    //~^ ERROR cannot find value `Into` in this scope
    //~| HELP use the path separator
}

macro_rules! Trait {
    () => {
        ::std::iter::Iterator
        //~^ ERROR cannot find value `Iterator` in module `::std::iter`
        //~| ERROR cannot find value `Iterator` in module `::std::iter`
    };
}

macro_rules! create {
    () => {
        Into::<String>.into("")
        //~^ ERROR cannot find value `Into` in this scope
        //~| HELP use the path separator
    };
}

fn interaction_with_macros() {
    //
    // Note that if the receiver is a macro call, we do not want to suggest to replace
    // `.` with `::` as that would be a syntax error.
    // Since the receiver is a trait and not a type, we cannot suggest to surround
    // it with angle brackets. It would be interpreted as a trait object type void of
    // `dyn` which is most likely not what the user intended to write.
    // `<_ as Trait!()>::` is also not an option as it's equally syntactically invalid.
    //

    Trait!().map(std::convert::identity); // no `help` here!

    Trait!().map; // no `help` here!

    //
    // Ensure that the suggestion is shown for expressions inside of macro definitions.
    //

    let _ = create!();
}
