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

// Regression test for https://github.com/rust-lang/rust/issues/155365
struct Demo {
    contents: Vec<u32>,
}

impl Demo {
    fn shared(&self) {
        self.contents.filter(|v| *v % 2 == 1).count(); //~ ERROR `Vec<u32>` is not an iterator
    }

    fn mutable(&mut self) {
        self.contents.filter(|v| *v % 2 == 1).count(); //~ ERROR `Vec<u32>` is not an iterator
    }

    fn owned(self) {
        self.contents.filter(|v| *v % 2 == 1).count(); //~ ERROR no method named `filter` found
    }
}

fn filter_param(contents: &Vec<u32>) {
    contents.filter(|v| *v % 2 == 1).count(); //~ ERROR no method named `filter` found
}

fn filter_explicit_deref(contents: &Vec<u32>) {
    (*contents).filter(|v| *v % 2 == 1).count(); //~ ERROR `Vec<u32>` is not an iterator
}
