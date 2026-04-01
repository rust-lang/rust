const fn foo<T>() {
    const { foo::<&T>() } //~ ERROR: queries overflow the depth limit!
}

fn main () {
    const X: () = foo::<i32>();
}
