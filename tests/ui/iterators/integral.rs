fn main() {
    for _ in 42 {}
    //~^ ERROR `{integer}` is not an iterator
    for _ in 42 as u8 {}
    //~^ ERROR `u8` is not an iterator
    for _ in 42 as i8 {}
    //~^ ERROR `i8` is not an iterator
    for _ in 42 as u16 {}
    //~^ ERROR `u16` is not an iterator
    for _ in 42 as i16 {}
    //~^ ERROR `i16` is not an iterator
    for _ in 42 as u32 {}
    //~^ ERROR `u32` is not an iterator
    for _ in 42 as i32 {}
    //~^ ERROR `i32` is not an iterator
    for _ in 42 as u64 {}
    //~^ ERROR `u64` is not an iterator
    for _ in 42 as i64 {}
    //~^ ERROR `i64` is not an iterator
    for _ in 42 as usize {}
    //~^ ERROR `usize` is not an iterator
    for _ in 42 as isize {}
    //~^ ERROR `isize` is not an iterator
    for _ in 42.0 {}
    //~^ ERROR `{float}` is not an iterator
}
