fn main() {
    let _ = &&[0] as &[_];
    //~^ ERROR non-primitive cast: `&&[i32; 1]` as `&[_]`
    let _ = 7u32 as Option<_>;
    //~^ ERROR non-primitive cast: `u32` as `Option<_>`
}
