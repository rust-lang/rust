// error-pattern: can only dereference structs
struct cat {
    x: ()
}

fn main() {
    let kitty : cat = cat { x: () };
    log (error, *kitty);
}
