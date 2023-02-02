// compile-flags: --crate-type=lib
// check-pass

// Make sure we don't pass inference variables to uninhabitedness checks in borrowck

struct Command<'s> {
    session: &'s (),
    imp: std::convert::Infallible,
}

fn command(_: &()) -> Command<'_> {
    unreachable!()
}

fn with_session<'s>(a: &std::process::Command, b: &'s ()) -> Command<'s> {
    a.get_program();
    command(b)
}
