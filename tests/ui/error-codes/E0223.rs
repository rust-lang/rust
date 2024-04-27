trait MyTrait { type X; }
struct MyStruct;
impl MyTrait for MyStruct {
    type X = ();
}

fn main() {
    let foo: MyTrait::X;
    //~^ ERROR ambiguous associated type
}
