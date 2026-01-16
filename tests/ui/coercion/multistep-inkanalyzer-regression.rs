//@ check-pass

// In the actual code from inkanalyzer, the code is more complex and non-recursive
#[allow(unconditional_recursion)]
fn ink_attrs_closest_descendants() -> impl Iterator<Item = ()> {
    if true {
        Box::new(ink_attrs_closest_descendants()) as Box<dyn Iterator<Item = ()>>
    } else {
        Box::new(ink_attrs_closest_descendants())
    }
}

fn main() {}
