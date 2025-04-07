// The original problem in #66340 was that `find_deprecation_generic`
// called `attr.meta().unwrap()` under the assumption that the attribute
// was a well-formed `MetaItem`.

fn main() {
    foo() //~ WARNING use of deprecated function `foo`
}

#[deprecated(note = test)]
//~^ ERROR expected unsuffixed literal, found `test`
fn foo() {}
