struct BadArraySize<const N: u8> {
    arr: [i32; N],
    //~^ ERROR the constant `N` is not of type `usize`
}

fn main() {
    let _ = BadArraySize::<2> { arr: [0, 0, 0] };
    //~^ ERROR mismatched types
    //~| ERROR the constant `2` is not of type `usize`
}

fn iter(val: BadArraySize::<2>) {
    for _ in val.arr {}
    //~^ ERROR the constant `2` is not of type `usize`
    //~| ERROR `[i32; 2]` is not an iterator
}

// issue #131102
pub struct Blorb<const N: u16>([String; N]); //~ ERROR the constant `N` is not of type `usize`
pub struct Wrap(Blorb<0>);
pub const fn i(_: Wrap) {} //~ ERROR destructor of `Wrap` cannot be evaluated at compile-time
