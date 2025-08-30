//@ run-rustfix
#![deny(unused_assignments)]
#![allow(unused_mut)]
struct Object;

fn change_object(mut object: &Object) { //~ HELP you might have meant to mutate
   let object2 = Object;
   object = object2; //~ ERROR mismatched types
}

fn change_object2(mut object: &Object) {
    //~^ HELP you might have meant to mutate
   let object2 = Object;
   object = &object2;
   //~^ ERROR `object2` does not live long enough
   //~| ERROR value assigned to `object` is never read
}

fn change_object3(mut object: &mut Object) {
    //~^ HELP you might have meant to mutate
    let object2 = Object; //~ HELP consider changing this to be mutable
    object = &mut object2;
    //~^ ERROR cannot borrow `object2` as mutable
    //~| ERROR value assigned to `object` is never read
}

fn main() {
    let mut object = Object;
    change_object(&mut object);
    change_object2(&mut object);
    change_object3(&mut object);
}
