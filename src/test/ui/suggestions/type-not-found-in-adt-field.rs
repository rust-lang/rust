struct S {
    m: Vec<Hashmap<String, ()>>, //~ ERROR cannot find type `Hashmap` in this scope
    //~^ NOTE not found in this scope
}
fn main() {}
