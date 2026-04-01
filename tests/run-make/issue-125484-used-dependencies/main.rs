pub type Foo = something::same::Thing;

mod something {
    pub mod same {
        pub struct Thing;
    }
}

fn main() {}
