
trait thing<A> {
    fn foo() -> Option<A>;
}
impl<A> int: thing<A> {
    fn foo() -> Option<A> { None }
}
fn foo_func<A, B: thing<A>>(x: B) -> Option<A> { x.foo() }

fn main() {

    for iter::eachi(Some({a: 0})) |i, a| { 
        #debug["%u %d", i, a.a];
    }

    let _x: Option<float> = foo_func(0);
}
