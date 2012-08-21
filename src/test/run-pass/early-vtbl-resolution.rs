
trait thing<A> {
    fn foo() -> option<A>;
}
impl<A> int: thing<A> {
    fn foo() -> option<A> { none }
}
fn foo_func<A, B: thing<A>>(x: B) -> option<A> { x.foo() }

fn main() {

    for iter::eachi(some({a: 0})) |i, a| { 
        #debug["%u %d", i, a.a];
    }

    let _x: option<float> = foo_func(0);
}
