fn main() {
    yield true; //~ ERROR yield syntax is experimental
                //~^ ERROR yield expression outside of coroutine literal
}

#[cfg(FALSE)]
fn foo() {
    yield; //~ ERROR yield syntax is experimental
    yield 0; //~ ERROR yield syntax is experimental
}
