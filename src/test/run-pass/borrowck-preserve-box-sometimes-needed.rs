// exec-env:RUST_POISON_ON_FREE=1

fn switcher(x: Option<@int>) {
    let mut x = x;
    match x {
      Some(@y) => { copy y; x = None; }
      None => { }
    }
}

fn main() {
    switcher(None);
    switcher(Some(@3));
}