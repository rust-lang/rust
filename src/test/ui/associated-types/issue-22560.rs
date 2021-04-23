trait Add<Rhs=Self> {
    type Output;
}

trait Sub<Rhs=Self> {
    type Output;
}

type Test = dyn Add + Sub;
//~^ ERROR E0393
//~| ERROR E0191
//~| ERROR E0393
//~| ERROR E0225

fn main() { }
