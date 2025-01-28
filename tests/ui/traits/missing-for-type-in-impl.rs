//@revisions: e2021 e2015
//@[e2021]edition: 2021
trait Foo<T> {
    fn id(me: T) -> T;
}

/* note the "missing" for ... (in this case for i64, in order for this to compile) */
impl Foo<i64> {
//[e2021]~^ ERROR expected a type, found a trait
//[e2015]~^^ WARNING trait objects without an explicit `dyn` are deprecated
//[e2015]~| WARNING trait objects without an explicit `dyn` are deprecated
//[e2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
//[e2015]~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!
//[e2015]~| ERROR the trait `Foo` is not dyn compatible
    fn id(me: i64) -> i64 {me}
}

fn main() {
    let x: i64 = <i64 as Foo<i64>>::id(10);
    //~^ ERROR the trait bound `i64: Foo<i64>` is not satisfied
    println!("{}", x);
}
