//@ check-pass
// Test for issue #151846: unused_allocation warning should ignore
// allocations to pass Box to things taking self: &Box

#![deny(unused_allocation)]

struct MyStruct;

trait TraitTakesBoxRef {
    fn trait_takes_box_ref(&self);
}

impl TraitTakesBoxRef for Box<MyStruct> {
    fn trait_takes_box_ref(&self) {}
}

impl MyStruct {
    fn inherent_takes_box_ref(self: &Box<Self>) {}
}

fn takes_box_ref(_: &Box<MyStruct>) {}

trait TraitTakesBoxVal {
    fn trait_takes_box_val(self);
}

impl TraitTakesBoxVal for Box<MyStruct> {
    fn trait_takes_box_val(self) {}
}

impl MyStruct {
    fn inherent_takes_box_val(self: Box<Self>) {}
}

fn takes_box_val(_: Box<MyStruct>) {}

pub fn foo() {
    // These should NOT warn - the allocation is necessary because
    // the method takes &Box<Self>
    Box::new(MyStruct).trait_takes_box_ref();
    Box::new(MyStruct).inherent_takes_box_ref();
    takes_box_ref(&Box::new(MyStruct));

    // These already don't warn - the allocation is necessary
    Box::new(MyStruct).trait_takes_box_val();
    Box::new(MyStruct).inherent_takes_box_val();
    takes_box_val(Box::new(MyStruct));

    // Fully-qualified syntax also does not warn:
    <Box<MyStruct> as TraitTakesBoxRef>::trait_takes_box_ref(&Box::new(MyStruct));
    MyStruct::inherent_takes_box_ref(&Box::new(MyStruct));
}

fn main() {}
