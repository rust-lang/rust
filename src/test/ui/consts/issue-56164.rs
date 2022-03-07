const fn foo() { (||{})() }
//~^ ERROR cannot call non-const closure

const fn bad(input: fn()) {
    input()
    //~^ ERROR function pointer
}

fn main() {
}
