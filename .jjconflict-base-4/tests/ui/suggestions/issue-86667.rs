// Regression test for #86667, where a garbled suggestion was issued for
// a missing named lifetime parameter.

//@ edition: 2018

async fn a(s1: &str, s2: &str) -> &str {
    //~^ ERROR: missing lifetime specifier [E0106]
    s1
}

fn b(s1: &str, s2: &str) -> &str {
    //~^ ERROR: missing lifetime specifier [E0106]
    s1
}

fn main() {}
