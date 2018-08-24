struct foo(usize);

fn main() {
    let (foo, _) = (2, 3); //~ ERROR let bindings cannot shadow tuple structs
}
