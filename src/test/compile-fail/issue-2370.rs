// error-pattern: type cat cannot be dereferenced
struct cat {
    foo: ()
}

fn main() {
    let nyan = cat { foo: () };
    log (error, *nyan);
}