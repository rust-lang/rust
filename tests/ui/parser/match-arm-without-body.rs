macro_rules! pat {
    () => { Some(_) }
}

fn main() {
    match Some(false) {
        Some(_)
    }
    match Some(false) {
        Some(_)
        _ => {}
        //~^ ERROR expected one of
    }
    match Some(false) {
        Some(_),
        //~^ ERROR unexpected `,` in pattern
        //~| HELP try adding parentheses to match on a tuple
        //~| HELP or a vertical bar to match on multiple alternatives
    }
    match Some(false) {
        Some(_),
        //~^ ERROR unexpected `,` in pattern
        //~| HELP try adding parentheses to match on a tuple
        //~| HELP or a vertical bar to match on multiple alternatives
        _ => {}
    }
    match Some(false) {
        Some(_) if true
    }
    match Some(false) {
        Some(_) if true
        _ => {}
        //~^ ERROR expected one of
    }
    match Some(false) {
        Some(_) if true,
    }
    match Some(false) {
        Some(_) if true,
        _ => {}
    }
    match Some(false) {
        pat!()
    }
    match Some(false) {
        pat!(),
    }
    match Some(false) {
        pat!() if true,
    }
    match Some(false) {
        pat!()
        //~^ ERROR expected `,` following `match` arm
        //~| HELP missing a comma here
        _ => {}
    }
    match Some(false) {
        pat!(),
        _ => {}
    }
}
