//@ edition:2018

fn main() {
    <_ as async Fn()>(|| async {});
    //~^ ERROR expected identifier, found keyword `async`
    //~| ERROR expected one of
}
