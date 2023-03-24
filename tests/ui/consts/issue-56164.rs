const fn foo() { (||{})() }
//~^ ERROR cannot call non-const closure
//~| ERROR the trait bound

const fn bad(input: fn()) {
    input()
    //~^ ERROR function pointer calls are not allowed
}

fn main() {
}
