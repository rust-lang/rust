// Tests that the compiler suggests an iterator method when an `Iterator` method
// is called on something that implements `IntoIterator`.

fn main() {
    let items = items();
    let other_items = items.map(|i| i + 1);
    //~^ ERROR no method named `map` found for opaque type `impl IntoIterator<Item = i32>` in the current scope
    //~| HELP: call `.into_iter()` first
    let vec: Vec<i32> = items.collect();
    //~^ ERROR no method named `collect` found for opaque type `impl IntoIterator<Item = i32>` in the current scope
    //~| HELP: call `.into_iter()` first
}

fn items() -> impl IntoIterator<Item = i32> {
    vec![1, 2, 3]
}

fn process(items: impl IntoIterator<Item = String>) -> Vec<String> {
    items.collect()
    //~^ ERROR no method named `collect` found for type parameter `impl IntoIterator<Item = String>` in the current scope
    //~| HELP: call `.into_iter()` first
}

// Regression test for https://github.com/rust-lang/rust/issues/155365
struct Demo {
    contents: Vec<u32>,
}

impl Demo {
    fn count_odds(&self) -> usize {
        self.contents.filter(|v| *v % 2 == 1).count()
        //~^ ERROR no method named `filter` found for struct `Vec<u32>` in the current scope
        //~| HELP: call `.iter()` first
    }

    fn increment(&mut self) {
        self.contents.for_each(|v| *v += 1)
        //~^ ERROR no method named `for_each` found for struct `Vec<u32>` in the current scope
        //~| HELP: call `.iter_mut()` first
    }
}

fn count_odds_param(contents: &Vec<u32>) -> usize {
    contents.filter(|v| *v % 2 == 1).count()
    //~^ ERROR no method named `filter` found for reference `&Vec<u32>` in the current scope
    //~| HELP: call `.into_iter()` first
}

fn count_odds_explicit_deref(contents: &Vec<u32>) -> usize {
    (*contents).filter(|v| *v % 2 == 1).count()
    //~^ ERROR no method named `filter` found for struct `Vec<u32>` in the current scope
    //~| HELP: call `.iter()` first
}
