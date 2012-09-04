type foo = Option<int>;

fn bar(_t: foo) {}

fn main() {
    // we used to print foo<int>:
    bar(Some(3u)); //~ ERROR mismatched types: expected `foo`
}