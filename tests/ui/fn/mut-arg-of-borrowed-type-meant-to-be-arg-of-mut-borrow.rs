#![deny(unused_assignments, unused_variables)]
struct Object;

fn change_object(mut object: &Object) {
    let object2 = Object;
    object = object2; //~ ERROR mismatched types
}

fn change_object2(mut object: &Object) { //~ ERROR variable `object` is assigned to, but never used
    let object2 = Object;
    object = &object2;
    //~^ ERROR `object2` does not live long enough
    //~| ERROR value assigned to `object` is never read
}

fn main() {
    let object = Object;
    change_object(&object);
    change_object2(&object);
}
