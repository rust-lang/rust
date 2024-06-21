fn function<T: PartialEq>() {
    foo == 2; //~ ERROR cannot find value `foo` [E0425]
}

fn main() {}
