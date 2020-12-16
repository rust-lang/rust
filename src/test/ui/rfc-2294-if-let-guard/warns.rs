#![feature(if_let_guard)]
#![allow(incomplete_features)]

#[deny(irrefutable_let_patterns)]
fn irrefutable_let_guard() {
    match Some(()) {
        Some(x) if let () = x => {}
        //~^ ERROR irrefutable if-let guard
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
