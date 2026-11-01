// rustc complains that the `else` clause does not diverge (even though it contains an
// unconditional panic)
//@no-rustfix
#![feature(try_blocks)]
#![expect(
    clippy::let_unit_value,
    clippy::never_loop,
    clippy::single_match,
    clippy::unused_unit
)]
#![warn(clippy::manual_let_else)]

fn g() -> Option<()> {
    None
}

fn main() {}

fn fire() {
    // The final expression will need to be turned into a statement.
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        panic!();
        ()
    };

    // Even if the result is buried multiple expressions deep.
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        panic!();
        if true {
            match 0 {
                0 => (),
                _ => (),
            }
        } else {
            panic!()
        }
    };

    // Or if a break gives the value.
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        loop {
            panic!();
            break ();
        }
    };

    // Even if the break is in a weird position.
    let v = if let Some(v_some) = g() {
        //~^ manual_let_else

        v_some
    } else {
        'a: loop {
            panic!();
            loop {
                match 0 {
                    0 if (return break 'a ()) => {},
                    _ => {},
                }
            }
        }
    };
}
