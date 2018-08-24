// Feature gate test for macro_at_most_once_rep under 2018 edition.

// gate-test-macro_at_most_once_rep
// edition:2018

macro_rules! foo {
    ($(a)?) => {}
    //~^ERROR using the `?` macro Kleene operator for
    //~|ERROR expected `*` or `+`
}

macro_rules! baz {
    ($(a),?) => {} //~ERROR expected `*` or `+`
}

macro_rules! barplus {
    ($(a)?+) => {}
    //~^ERROR using the `?` macro Kleene operator for
    //~|ERROR expected `*` or `+`
}

macro_rules! barstar {
    ($(a)?*) => {}
    //~^ERROR using the `?` macro Kleene operator for
    //~|ERROR expected `*` or `+`
}

pub fn main() {
    foo!();
    foo!(a);
    foo!(a?); //~ ERROR no rules expected the token `?`
    foo!(a?a); //~ ERROR no rules expected the token `?`
    foo!(a?a?a); //~ ERROR no rules expected the token `?`
}

