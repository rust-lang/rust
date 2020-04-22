use std::sync::mpsc;

fn main() {
    let (tx, rx) = mpsc::channel::new(1);
    //~^ ERROR `channel` in `mpsc` is a concrete value, not a module or Struct you specified
}
