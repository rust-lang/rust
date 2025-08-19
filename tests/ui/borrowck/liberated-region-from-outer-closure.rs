// Regression test for <https://github.com/rust-lang/rust/issues/144608>.

fn example<T: Copy>(x: T) -> impl FnMut(&mut ()) {
    move |_: &mut ()| {
        move || needs_static_lifetime(x);
        //~^ ERROR the parameter type `T` may not live long enough
    }
}

fn needs_static_lifetime<T: 'static>(obj: T) {}

fn main() {}
