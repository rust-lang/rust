fn main() {
    &1 as Send; //~ ERROR cast to unsized type
    Box::new(1) as Send; //~ ERROR cast to unsized type
}
