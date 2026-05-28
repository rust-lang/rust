fn main() {
    &1 as dyn Send; //~ ERROR cast to unsized
    Box::new(1) as dyn Send; //~ ERROR cast to unsized
}
