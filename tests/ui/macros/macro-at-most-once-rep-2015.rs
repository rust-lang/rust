// Tests that `?` is a Kleene op and not a macro separator in the 2015 edition.

//@ edition:2015

macro_rules! foo {
    ($(a)?) => {};
}

// The Kleene op `?` does not admit a separator before it.
macro_rules! baz {
    ($(a),?) => {}; //~ERROR the `?` macro repetition operator
}

macro_rules! barplus {
    ($(a)?+) => {}; // ok. matches "a+" and "+"
}

macro_rules! barstar {
    ($(a)?*) => {}; // ok. matches "a*" and "*"
}

pub fn main() {
    foo!();
    foo!(a);
    foo!(a?); //~ ERROR no rules expected `?`
    foo!(a?a); //~ ERROR no rules expected `?`
    foo!(a?a?a); //~ ERROR no rules expected `?`

    barplus!(); //~ERROR unexpected end of macro invocation
    barplus!(a); //~ERROR unexpected end of macro invocation
    barplus!(a?); //~ ERROR no rules expected `?`
    barplus!(a?a); //~ ERROR no rules expected `?`
    barplus!(a+);
    barplus!(+);

    barstar!(); //~ERROR unexpected end of macro invocation
    barstar!(a); //~ERROR unexpected end of macro invocation
    barstar!(a?); //~ ERROR no rules expected `?`
    barstar!(a?a); //~ ERROR no rules expected `?`
    barstar!(a*);
    barstar!(*);
}
