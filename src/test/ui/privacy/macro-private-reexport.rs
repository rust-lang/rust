// edition:2021

mod foo {
    macro_rules! bar {
        () => {};
    }

    pub use bar as _; //~ ERROR `bar` is only public within the crate, and cannot be re-exported outside
}

fn main() {}
