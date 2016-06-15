#![feature(start)]

#[start]
fn foo(nargs: isize, args: *const *const u8) -> isize {
    if nargs > 0 {
        assert!(unsafe{*args} as usize != 0);
    }
    0
}
