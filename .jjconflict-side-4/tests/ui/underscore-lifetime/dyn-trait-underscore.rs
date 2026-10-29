// Check that the `'_` in `dyn Trait + '_` acts like ordinary elision,
// and not like an object lifetime default.
//
// cc #48468

fn a<T>(items: &[T]) -> Box<dyn Iterator<Item=&T>> {
    //                      ^^^^^^^^^^^^^^^^^^^^^ bound *here* defaults to `'static`
    Box::new(items.iter())
    //~^ ERROR lifetime may not live long enough
}

fn b<T>(items: &[T]) -> Box<dyn Iterator<Item=&T> + '_> {
    Box::new(items.iter()) // OK, equivalent to c
}

fn c<'a, T>(items: &'a [T]) -> Box<dyn Iterator<Item=&'a T> + 'a> {
    Box::new(items.iter()) // OK, equivalent to b
}

fn main() { }
