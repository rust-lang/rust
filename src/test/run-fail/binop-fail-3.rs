// error-pattern:quux
fn foo() -> ! {
    panic!("quux");
}

#[allow(resolve_trait_on_defaulted_unit)]
fn main() {
    foo() == foo(); // these types wind up being defaulted to ()
}
