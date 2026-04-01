//@ edition:2015

#[macro_export]
macro_rules! generate_trait_2015_ident {
    ($Type: ident) => {
        trait Trait1 {
            fn method($Type) {}
        }
    };
}

#[macro_export]
macro_rules! generate_trait_2015_tt {
    ($Type: tt) => {
        trait Trait2 {
            fn method($Type) {}
        }
    };
}

fn main() {}
