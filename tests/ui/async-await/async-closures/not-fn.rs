//@ edition:2021

// FIXME(async_closures): This needs a better error message!

fn main() {
    fn needs_fn<T>(_: impl FnMut() -> T) {}

    let mut x = 1;
    needs_fn(async || {
        //~^ ERROR async closure does not implement `FnMut` because it captures state from its environment
        x += 1;
    });
}
