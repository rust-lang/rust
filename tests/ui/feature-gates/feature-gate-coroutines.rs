// revisions: e2024 none
//[e2024] compile-flags: --edition 2024 -Zunstable-options

fn main() {
    yield true; //~ ERROR yield syntax is experimental
                //~^ ERROR yield expression outside of coroutine literal
                //[none]~^^ ERROR yield syntax is experimental

    let _ = || yield true; //~ ERROR yield syntax is experimental
    //[none]~^ ERROR yield syntax is experimental
}

#[cfg(FALSE)]
fn foo() {
    // Ok in 2024 edition
    yield; //[none]~ ERROR yield syntax is experimental
    yield 0; //[none]~ ERROR yield syntax is experimental
}
