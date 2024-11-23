struct BadArraySize<const N: u8> {
    arr: [i32; N],
    //~^ ERROR the constant `N` is not of type `usize`
}

fn main() {
    let _ = BadArraySize::<2> { arr: [0, 0, 0] };
    //~^ ERROR mismatched types
    //~| ERROR the constant `2` is not of type `usize`
}
