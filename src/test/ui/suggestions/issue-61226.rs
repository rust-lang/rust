struct X {}
fn f() {
    vec![X]; //…
    //~^ ERROR expected value, found struct `X`
}
