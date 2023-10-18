struct MyStruct {
    pub s1: Option<String>,
}

fn main() {
    let thing = MyStruct { s1: None };

    match thing {
        MyStruct { .., Some(_) } => {},
        //~^ ERROR missing field name before pattern
        //~| ERROR expected `}`, found `,`
        _ => {}
    }
}
