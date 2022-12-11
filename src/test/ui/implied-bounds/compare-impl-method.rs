// Check implied bounds are used when comparing trait and impl methods.
// issue: #105495
// check-pass

trait Trait {
    fn get();
}

// An implied bound 'b: 'a
impl<'a, 'b> Trait for &'a &'b u8 {
    fn get() where 'b: 'a, {}
}

// An explicit bound 'b: 'a
impl<'a, 'b> Trait for (&'a u8, &'b u8) where 'b: 'a, {
    fn get() where 'b: 'a, {}
}

fn main() {}
