//@ run-rustfix

fn main() {
    mut let _x = 123;
    //~^ ERROR invalid variable declaration
}
