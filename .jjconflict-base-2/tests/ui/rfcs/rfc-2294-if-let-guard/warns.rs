#[deny(irrefutable_let_patterns)]
fn irrefutable_let_guard() {
    match Some(()) {
        Some(x) if let () = x => {}
        //~^ ERROR irrefutable `if let` guard
        _ => {}
    }
}

#[deny(irrefutable_let_patterns)]
fn trailing_irrefutable_pattern_binding() {
    match Some(5) {
        o if let x = 0 => {}
        //~^ ERROR irrefutable `if let` guard
        _ => {}
    }
}

#[deny(irrefutable_let_patterns)]
fn trailing_irrefutable_in_let_chain() {
    match Some(5) {
        Some(x) if let Some(y) = Some(x) && let z = 0 => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
fn unreachable_pattern() {
    match Some(()) {
        x if let None | None = x => {}
        //~^ ERROR unreachable pattern
        _ => {}
    }
}

fn main() {}
