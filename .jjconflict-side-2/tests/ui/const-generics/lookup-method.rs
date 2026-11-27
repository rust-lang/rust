// https://github.com/rust-lang/rust/issues/124946

struct Builder<const A: bool, const B: bool>;

impl<const A: bool> Builder<A, false> {
    fn cast(self) -> Builder<A, true> {
        Builder
    }
}

impl Builder<true, true> {
    fn build(self) {}
}

fn main() {
    let b = Builder::<false, false>;
    b.cast().build();
    //~^ ERROR: no method named `build` found for struct `Builder<false, true>` in the current scope
}
