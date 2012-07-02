type foo = option<int>;

fn bar(_t: foo) {}

fn main() {
    // we used to print foo<int>:
    bar(some(3u)); //~ ERROR mismatched types: expected `foo`
}