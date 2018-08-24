fn f<Z>() -> bool {
    enum E { V(Z) }
    //~^ ERROR can't use type parameters from outer function
    true
}

fn main() {
    let b = f::<isize>();
    assert!(b);
}
