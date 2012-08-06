// exec-env:RUST_POISON_ON_FREE=1

fn switcher(x: option<@int>) {
    let mut x = x;
    match x {
      some(@y) => { copy y; x = none; }
      none => { }
    }
}

fn main() {
    switcher(none);
    switcher(some(@3));
}