// Tests that the compiler suggests an `into_iter` call when an `Iterator` method
// is called on something that implements `IntoIterator`

fn main() {
    let items = items();
    let other_items = items.map(|i| i + 1);
    //~^ ERROR no method named `map` found for opaque type `impl IntoIterator<Item = i32>` in the current scope
    let vec: Vec<i32> = items.collect();
    //~^ ERROR no method named `collect` found for opaque type `impl IntoIterator<Item = i32>` in the current scope
}

fn items() -> impl IntoIterator<Item = i32> {
    vec![1, 2, 3]
}

fn process(items: impl IntoIterator<Item = String>) -> Vec<String> {
    items.collect()
    //~^ ERROR no method named `collect` found for type parameter `impl IntoIterator<Item = String>` in the current scope
}
