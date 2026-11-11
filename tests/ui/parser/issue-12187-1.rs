fn new<T>() -> &'static T {
    panic!()
}

fn main() {
    let &v = new();
    //~^ ERROR type annotations needed
}
