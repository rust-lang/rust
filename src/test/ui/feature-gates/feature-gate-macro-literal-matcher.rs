// Test that the :lifetime macro fragment cannot be used when macro_lifetime_matcher
// feature gate is not used.

macro_rules! m { ($lt:literal) => {} }
//~^ ERROR :literal fragment specifier is experimental and subject to change

fn main() {
    m!("some string literal");
}
