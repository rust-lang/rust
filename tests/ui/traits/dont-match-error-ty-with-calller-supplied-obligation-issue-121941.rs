fn function<T: PartialEq>() {
    foo == 2; //~ ERROR cannot find value `foo`
}

fn main() {}
