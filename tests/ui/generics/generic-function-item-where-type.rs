fn foo<U>() {}

fn main() {
    foo::<main>()
    //~^ ERROR constant provided when a type was expected
}
