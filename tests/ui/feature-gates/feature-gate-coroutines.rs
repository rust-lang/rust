//@ revisions: e2024 none
//@[e2024] edition: 2024

fn main() {
    true.yield; //~ ERROR yield syntax is experimental
                //~^ ERROR yield expression outside of coroutine literal
                //~^^ ERROR yield syntax is experimental
                //~^^^ ERROR `yield` can only be used

    let _ = || true.yield; //~ ERROR yield syntax is experimental
    //~^ ERROR yield syntax is experimental
    //~^^ ERROR `yield` can only be used
}

#[cfg(false)]
fn foo() {
    // Ok in 2024 edition
    ().yield; //~ ERROR yield syntax is experimental
    0.yield; //~ ERROR yield syntax is experimental
}
