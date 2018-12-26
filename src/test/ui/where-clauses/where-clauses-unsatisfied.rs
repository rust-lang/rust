fn equal<T>(a: &T, b: &T) -> bool where T : Eq { a == b }

struct Struct;

fn main() {
    drop(equal(&Struct, &Struct))
    //~^ ERROR the trait bound `Struct: std::cmp::Eq` is not satisfied
}
