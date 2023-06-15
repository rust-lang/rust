// run-pass
#![feature(explicit_tail_calls)]
use std::panic::Location;

fn main() {
    assert_eq!(get_caller_location().line(), 6);
    assert_eq!(get_caller_location().line(), 7);
}

#[track_caller]
fn get_caller_location() -> &'static Location<'static> {
    #[track_caller]
    fn inner() -> &'static Location<'static> {
        become Location::caller()
    }

    become inner()
}
