//@ build-pass
//@ edition: 2021

// Demonstrates that an async closure may implement `FnMut` (not just `AsyncFnMut`!)
// if it has no self-borrows. In this case, `&Ty` is not borrowed from the closure env,
// since it's fine to reborrow it with its original lifetime. See the doc comment on
// `should_reborrow_from_env_of_parent_coroutine_closure` for more detail for when we
// must borrow from the closure env.

fn main() {
    hello(&Ty);
}

fn needs_fn_mut<T>(mut x: impl FnMut() -> T) {
    x();
}

fn hello(x: &Ty) {
    needs_fn_mut(async || { x.hello(); });
}

struct Ty;
impl Ty {
    fn hello(&self) {}
}
