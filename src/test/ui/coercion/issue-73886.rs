fn main() {
    let _ = 7u32 as Option<_>;
    //~^ ERROR non-primitive cast: `u32` as `Option<_>`
}
