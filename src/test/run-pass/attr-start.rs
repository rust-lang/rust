// pretty-expanded FIXME #23616

#![feature(start)]

#[start]
fn start(_argc: isize, _argv: *const *const u8) -> isize {
    return 0;
}
