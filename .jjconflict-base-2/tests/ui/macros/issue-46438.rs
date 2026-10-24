macro_rules! m {
    ($my_type: ty) => {
        impl $my_type for u8 {}
    }
}

trait Tr {}

m!(Tr);

m!(&'static u8); //~ ERROR expected a trait, found type

fn main() {}
