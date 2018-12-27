enum Option<T> {
    None,
    Some(T),
}

fn main() {
    match &mut Some(1) {
        ref mut z @ &mut Some(ref a) => {
        //~^ ERROR pattern bindings are not allowed after an `@`
            **z = None;
            println!("{}", *a);
        }
        _ => ()
    }
}
