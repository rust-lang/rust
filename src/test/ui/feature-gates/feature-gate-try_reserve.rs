fn main() {
    let v = Vec::new();
    v.try_reserve(10); //~ ERROR: use of unstable library feature 'try_reserve'
}
