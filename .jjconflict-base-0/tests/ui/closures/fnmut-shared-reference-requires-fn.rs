//! Do not suggest a mutable reference when the root obligation genuinely requires `Fn`.

fn requires_fn<F: Fn()>(_: F) {}

fn main() {
    let mut value = 0;
    let mut func = || value += 1;
    //~^ ERROR expected a closure that implements the `Fn` trait

    requires_fn(&func);
}
