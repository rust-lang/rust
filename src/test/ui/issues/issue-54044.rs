#[cold] //~ ERROR attribute should be applied to a function
struct Foo; //~ NOTE not a function

fn main() {
    #[cold] //~ ERROR attribute should be applied to a function
    5; //~ NOTE not a function
}
