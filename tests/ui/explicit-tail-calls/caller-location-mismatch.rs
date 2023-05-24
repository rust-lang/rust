#![feature(explicit_tail_calls)]
use std::panic::Location;

fn get_some_location1() -> &'static Location<'static> {
    #[track_caller]
    fn inner() -> &'static Location<'static> {
        become Location::caller()
    }

    become inner()
    //~^ error: a function mot marked with `#[track_caller]` cannot tail-call one that is
}

#[track_caller]
fn get_some_location2() -> &'static Location<'static> {
    fn inner() -> &'static Location<'static> {
        become Location::caller()
        //~^ error: a function mot marked with `#[track_caller]` cannot tail-call one that is
    }

    become inner()
    //~^ error: a function marked with `#[track_caller]` cannot tail-call one that is not
}

fn main() {
    get_some_location1();
    get_some_location2();
}
