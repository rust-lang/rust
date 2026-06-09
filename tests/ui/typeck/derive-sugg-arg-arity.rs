pub struct A;

fn main() {
    match () {
        _ => match A::partial_cmp() {},
        //~^ ERROR the associated function or constant `partial_cmp` exists for struct `A`, but its trait bounds were not satisfied
    }
}
