// error-pattern: can only dereference structs
struct cat {
    foo: ()
}

fn main() {
    let nyan = cat { foo: () };
    log (error, *nyan);
}
