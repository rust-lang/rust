trait MyTrait { type X; }

fn main() {
    let foo: MyTrait::X;
    //~^ ERROR ambiguous associated type
}
