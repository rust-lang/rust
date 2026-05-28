fn foo<T>() -> T { panic!("Rocks for my pillow") }

fn main() {
    let x = match () { //~ ERROR type annotations needed
        () => foo() // T here should be unresolved
    };
}
