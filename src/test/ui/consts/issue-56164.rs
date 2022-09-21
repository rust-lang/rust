const fn foo() { (||{})() }
//~^ ERROR cannot call non-const closure
//~| ERROR erroneous constant
//~| WARN this was previously accepted

const fn bad(input: fn()) {
    input()
    //~^ ERROR function pointer calls are not allowed
}

fn main() {
}
