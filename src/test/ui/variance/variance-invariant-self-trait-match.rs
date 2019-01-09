#![allow(dead_code)]

trait Get {
    fn get(&self) -> Self;
}

fn get_min_from_max<'min, 'max, G>()
    where 'max : 'min, &'max G : Get, G : 'max
{
    impls_get::<&'min G>(); //~ ERROR mismatched types
}

fn get_max_from_min<'min, 'max, G>()
    where 'max : 'min, &'min G : Get, G : 'min
{
    impls_get::<&'max G>(); //~ ERROR mismatched types
}

fn impls_get<G>() where G : Get { }

fn main() { }
