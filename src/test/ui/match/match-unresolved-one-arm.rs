fn foo<T>() -> T { panic!("Rocks for my pillow") }

fn main() {
    let x = match () {
        () => foo() //~ ERROR type annotations needed
    };
}
