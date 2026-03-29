// compile-fail

fn main() {
    "".parse();
    //~^ ERROR type annotations needed
    //~| HELP consider specifying a concrete type for the generic type `F`
}
