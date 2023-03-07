fn main() {
    vec![true, false].map(|v| !v).collect::<Vec<_>>();
    //~^ ERROR `Vec<bool>` is not an iterator
}
