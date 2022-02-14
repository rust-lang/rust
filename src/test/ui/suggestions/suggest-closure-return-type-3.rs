fn unbound_drop(_: impl Sized) {}

fn main() {
    unbound_drop(|| []);
    //~^ ERROR type annotations needed for `[_; 0]`
    //~| HELP try giving this closure an explicit return type
}
