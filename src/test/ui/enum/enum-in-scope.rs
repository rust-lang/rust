struct hello(isize);

fn main() {
    let hello = 0; //~ERROR let bindings cannot shadow tuple structs
}
