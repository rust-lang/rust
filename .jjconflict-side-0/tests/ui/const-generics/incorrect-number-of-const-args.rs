fn foo<const X: usize, const Y: usize>() -> usize {
    0
}

fn main() {
    foo::<0>();
    //~^ ERROR function takes 2

    foo::<0, 0, 0>();
    //~^ ERROR function takes 2
}
