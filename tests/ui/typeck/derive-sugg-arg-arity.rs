pub struct A;

fn main() {
    match () {
        _ => match A::partial_cmp() {},
        //~^ ERROR `A` is not an iterator
    }
}
