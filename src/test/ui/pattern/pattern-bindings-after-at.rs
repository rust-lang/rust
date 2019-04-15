enum Option<T> {
    None,
    Some(T),
}

fn main() {
    match &mut Some(1) {
        ref mut z @ &mut Some(ref a) => {
        //~^ ERROR pattern bindings are not allowed after an `@`
        //~| WARN cannot borrow `_` as immutable because it is also borrowed as mutable
        //~| WARN this error has been downgraded to a warning for backwards compatibility
        //~| WARN this represents potential undefined behavior in your code and this warning will
            **z = None;
            println!("{}", *a);
        }
        _ => ()
    }
}
