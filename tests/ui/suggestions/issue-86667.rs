// Regression test for #86667, where a garbled suggestion was issued for
// a missing named lifetime parameter.

// compile-flags: --edition 2018

async fn a(s1: &str, s2: &str) -> &str {
    //~^ ERROR: missing lifetime specifier [E0106]
    s1
    //~^ ERROR: lifetime may not live long enough
}

fn b(s1: &str, s2: &str) -> &str {
    //~^ ERROR: missing lifetime specifier [E0106]
    s1
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
