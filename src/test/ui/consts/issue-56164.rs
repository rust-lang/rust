const fn foo() { (||{})() }
//~^ ERROR the trait bound

const fn bad(input: fn()) {
    input()
}

fn main() {
}
