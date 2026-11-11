struct BuildData {
    foo: isize,
    foo: isize,
    //~^ ERROR field `foo` is already declared [E0124]
}

fn main() {
}
