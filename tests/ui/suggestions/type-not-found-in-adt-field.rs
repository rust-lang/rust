struct Struct {
    m: Vec<Someunknownname<String, ()>>, //~ ERROR cannot find type `Someunknownname` in this scope
    //~^ NOTE not found in this scope
}
struct OtherStruct { //~ HELP you might be missing a type parameter
    m: K, //~ ERROR cannot find type `K` in this scope
    //~^ NOTE not found in this scope
}
fn main() {}
