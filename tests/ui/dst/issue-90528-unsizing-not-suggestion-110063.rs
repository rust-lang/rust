trait Test {}
impl Test for &[u8] {}

fn needs_test<T: Test>() -> T {
    panic!()
}

fn main() {
    needs_test::<[u8; 1]>();
    //~^ ERROR the trait
    let x: [u8; 1] = needs_test();
    //~^ ERROR the trait
}
