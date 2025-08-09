//@ edition: 2024

mod ciallo {
    macro_rules! foo { () => {}; }
    pub use foo;
         //~^ERROR `foo` is only public within the crate, and cannot be re-exported outside
}

fn main() {}
