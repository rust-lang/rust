// While `feature(inline_const_pat)` has been removed from the
// compiler, we should still make sure that the resulting error
// message is acceptable.
fn main() {
    match 1 {
        const { 1 + 7 } => {}
        //~^ ERROR arbitrary expressions aren't allowed in patterns
        2 => {}
        _ => {}
    }

    match 5 {
        const { 1 } ..= 10 => {}
        //~^ ERROR arbitrary expressions aren't allowed in patterns
        _ => {}
    }

    match 5 {
        1 ..= const { 10 } => {}
        //~^ ERROR arbitrary expressions aren't allowed in patterns
        _ => {}
    }

    match 5 {
        const { 1 } ..= const { 10 } => {}
        //~^ ERROR arbitrary expressions aren't allowed in patterns
        //~| ERROR arbitrary expressions aren't allowed in patterns
        _ => {}
    }

    match 5 {
        const { 1 } .. 10 => {}
        //~^ ERROR arbitrary expressions aren't allowed in patterns
        _ => {}
    }

    match 5 {
        1 .. const { 10 } => {}
        //~^ ERROR arbitrary expressions aren't allowed in patterns
        _ => {}
    }

    match 5 {
        const { 1 + 2 } ..= 10 => {}
        //~^ ERROR arbitrary expressions aren't allowed in patterns
        _ => {}
    }

    match 5 {
        1 ..= const { 5 + 5 } => {}
        //~^ ERROR arbitrary expressions aren't allowed in patterns
        _ => {}
    }

    match 5 {
        const { 3 } .. => {}
        //~^ ERROR arbitrary expressions aren't allowed in patterns
        _ => {}
    }

    match 5 {
        ..= const { 7 } => {}
        //~^ ERROR arbitrary expressions aren't allowed in patterns
        _ => {}
    }
}
