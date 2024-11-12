//@ revisions: e2024 none
//@[e2024] compile-flags: --edition 2024 -Zunstable-options

fn main() {
    yield true; //~ ERROR yield syntax is experimental
                //~^ ERROR yield expression outside of coroutine literal
                //~^^ ERROR yield syntax is experimental
                //~^^^ ERROR `yield` can only be used

    let _ = || yield true; //~ ERROR yield syntax is experimental
    //~^ ERROR yield syntax is experimental
    //~^^ ERROR `yield` can only be used
}

#[cfg(FALSE)]
fn foo() {
    // Ok in 2024 edition
    yield; //~ ERROR yield syntax is experimental
    yield 0; //~ ERROR yield syntax is experimental
}
