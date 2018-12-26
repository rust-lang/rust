struct MyStruct {
    pub s1: Option<String>,
}

fn main() {
    let thing = MyStruct { s1: None };

    match thing {
        MyStruct { .., Some(_) } => {}, //~ ERROR pattern does not mention field `s1`
        //~^ ERROR expected `,`
        //~| ERROR expected `}`, found `,`
        _ => {}
    }
}
