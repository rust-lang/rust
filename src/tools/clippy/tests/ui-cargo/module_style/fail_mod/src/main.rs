#![warn(clippy::self_named_module_files)]

mod bad;

fn main() {
    let _ = bad::Thing;
    let _ = bad::inner::stuff::Inner;
    let _ = bad::inner::stuff::most::Snarks;
}
