mod MyMod {}

fn main() {
    let myVar = MyMod { T: 0 };
    //~^ ERROR expected struct, variant or union type, found module `MyMod`
}
