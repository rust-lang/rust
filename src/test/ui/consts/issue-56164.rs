const fn foo() { (||{})() }
//~^ ERROR cannot call non-const closure
//~| ERROR erroneous constant used [const_err]
//~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

const fn bad(input: fn()) {
    input()
    //~^ ERROR function pointer
}

fn main() {
}
