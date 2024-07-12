struct Struct {
    m: Vec<Someunknownname<String, ()>>, //~ ERROR cannot find type `Someunknownname`
    //~^ NOTE not found
}
struct OtherStruct { //~ HELP you might be missing a type parameter
    m: K, //~ ERROR cannot find type `K`
    //~^ NOTE not found
}
fn main() {}
