//@ edition:2018

async fn do_the_thing() -> u8 {
    8
}
// #63398: point at the enclosing scope and not the previously seen closure
fn main() {  //~ NOTE this is not `async`
    let x = move || {};
    let y = do_the_thing().await; //~ ERROR `await` is only allowed inside `async` functions
    //~^ NOTE only allowed inside `async` functions and blocks
}
