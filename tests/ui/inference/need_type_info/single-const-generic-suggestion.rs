// compile-fail

fn func<const CONST: usize>() {}

fn main() {
    func();
    //~^ ERROR type annotations needed
    //~| HELP consider specifying a const for the const generic `CONST`
}
