#![allow(dead_code)]

// Test that even when `Self` is only used in contravariant position, it
// is treated as invariant.

trait Get {
    fn get(&self);
}

fn get_min_from_max<'min, 'max, G>()
    where 'max : 'min, G : 'max, &'max G : Get
{
    impls_get::<&'min G>(); //~ ERROR mismatched types
}

fn get_max_from_min<'min, 'max, G>()
    where 'max : 'min, G : 'max, &'min G : Get
{
    // Previously OK, but now error because traits are invariant with
    // respect to all inputs.

    impls_get::<&'max G>(); //~ ERROR mismatched types
}

fn impls_get<G>() where G : Get { }

fn main() { }
