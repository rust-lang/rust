#![feature(start)]

#[start]
fn foo(_nargs: isize, _args: *const *const u8) -> isize {
    return 0;
}
