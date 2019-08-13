struct X {}
fn main() {
    vec![X]; //…
    //~^ ERROR expected value, found struct `X`
}
