// edition:2015

#[macro_export]
macro_rules! generate_trait_2015 {
    ($Type: ident) => {
        trait Trait {
            fn method($Type) {}
        }
    };
}

fn main() {}
