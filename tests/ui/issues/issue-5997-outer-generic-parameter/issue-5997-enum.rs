fn f<Z>() -> bool {
    enum E { V(Z) }
    //~^ ERROR can't use generic parameters from outer item
    true
}

fn main() {
    let b = f::<isize>();
    assert!(b);
}
