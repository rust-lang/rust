#![feature(start)]

#[start]
#[track_caller] //~ ERROR `start` is not allowed to be `#[track_caller]`
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    panic!("{}: oh no", std::panic::Location::caller());
}
