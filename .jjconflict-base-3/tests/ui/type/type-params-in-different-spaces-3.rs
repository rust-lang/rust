trait Tr : Sized {
    fn test<X>(u: X) -> Self {
        u   //~ ERROR mismatched types
    }
}

fn main() {}
