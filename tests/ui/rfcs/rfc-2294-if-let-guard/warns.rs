//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

#[deny(irrefutable_let_patterns)]
fn irrefutable_let_guard() {
    match Some(()) {
        Some(x) if let () = x => {}
        //~^ ERROR irrefutable `if let` guard
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
