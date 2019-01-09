enum Homura {
    Akemi { madoka: () }
}

fn main() {
    let homura = Homura::Akemi { kaname: () };
    //~^ ERROR variant `Homura::Akemi` has no field named `kaname`
}
