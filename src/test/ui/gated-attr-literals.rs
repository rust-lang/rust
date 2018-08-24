// Check that literals in attributes don't parse without the feature gate.

// gate-test-attr_literals

#![feature(custom_attribute)]

#[fake_attr] // OK
#[fake_attr(100)]
    //~^ ERROR non-string literals in attributes
#[fake_attr(1, 2, 3)]
    //~^ ERROR non-string literals in attributes
#[fake_attr("hello")]
    //~^ ERROR string literals in top-level positions, are experimental
#[fake_attr(name = "hello")] // OK
#[fake_attr(1, "hi", key = 12, true, false)]
    //~^ ERROR non-string literals in attributes, or string literals in top-level positions
#[fake_attr(key = "hello", val = 10)]
    //~^ ERROR non-string literals in attributes
#[fake_attr(key("hello"), val(10))]
    //~^ ERROR non-string literals in attributes, or string literals in top-level positions
#[fake_attr(enabled = true, disabled = false)]
    //~^ ERROR non-string literals in attributes
#[fake_attr(true)]
    //~^ ERROR non-string literals in attributes
#[fake_attr(pi = 3.14159)]
    //~^ ERROR non-string literals in attributes
#[fake_attr(b"hi")]
    //~^ ERROR string literals in top-level positions, are experimental
#[fake_doc(r"doc")]
    //~^ ERROR string literals in top-level positions, are experimental
struct Q {  }

fn main() { }
