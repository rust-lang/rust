//@ check-pass
//@ edition: 2021

// Demonstrates that an async closure may implement `FnMut` (not just `async FnMut`!)
// if it has no self-borrows. In this case, `&Ty` is not borrowed from the closure env,
// since it's fine to reborrow it with its original lifetime. See the doc comment on
// `should_reborrow_from_env_of_parent_coroutine_closure` for more detail for when we
// must borrow from the closure env.

#![feature(async_closure)]

fn main() {}

fn needs_fn_mut<T>(x: impl FnMut() -> T) {}

fn hello(x: &Ty) {
    needs_fn_mut(async || { x.hello(); });
}

struct Ty;
impl Ty {
    fn hello(&self) {}
}
