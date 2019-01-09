#![allow(dead_code)]

// Test that even when `Self` is only used in covariant position, it
// is treated as invariant.

trait Get {
    fn get() -> Self;
}

fn get_min_from_max<'min, 'max, G>()
    where 'max : 'min, G : 'max, &'max G : Get
{
    // Previously OK, now an error as traits are invariant.
    impls_get::<&'min G>(); //~ ERROR mismatched types
}

fn get_max_from_min<'min, 'max, G>()
    where 'max : 'min, G : 'max, &'min G : Get
{
    impls_get::<&'max G>(); //~ ERROR mismatched types
}

fn impls_get<G>() where G : Get { }

fn main() { }
