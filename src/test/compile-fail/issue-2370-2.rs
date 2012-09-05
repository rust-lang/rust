// error-pattern: type cat cannot be dereferenced
struct cat {
    x: ()
}

fn main() {
    let kitty : cat = cat { x: () };
    log (error, *kitty);
}