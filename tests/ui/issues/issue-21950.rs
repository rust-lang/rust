trait Add<Rhs=Self> {
    type Output;
}

impl Add for i32 {
    type Output = i32;
}

fn main() {
    let x = &10 as &dyn Add;
    //~^ ERROR E0393
    //~| ERROR E0191
}
