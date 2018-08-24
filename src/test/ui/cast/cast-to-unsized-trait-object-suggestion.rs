fn main() {
    &1 as Send; //~ ERROR cast to unsized
    Box::new(1) as Send; //~ ERROR cast to unsized
}
