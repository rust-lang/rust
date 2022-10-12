#[collapse_debuginfo]
//~^ ERROR the `#[collapse_debuginfo]` attribute is an experimental feature
macro_rules! foo {
    ($e:expr) => { $e }
}

fn main() {}
