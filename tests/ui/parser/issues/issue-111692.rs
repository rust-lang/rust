mod module {
    #[derive(Eq, PartialEq)]
    pub struct Type {
        pub x: u8,
        pub y: u8,
    }

    pub const C: u8 = 32u8;
}

fn test(x: module::Type) {
    if x == module::Type { x: module::C, y: 1 } { //~ ERROR struct literals are not allowed here
    }
}

fn test2(x: module::Type) {
    if x ==module::Type { x: module::C, y: 1 } { //~ ERROR struct literals are not allowed here
    }
}


fn test3(x: module::Type) {
    use module::Type;
    if x == Type { x: module::C, y: 1 } { //~ ERROR struct literals are not allowed here
    }
}

fn test4(x: module::Type) {
    use module as demo_module;
    if x == demo_module::Type { x: module::C, y: 1 } { //~ ERROR struct literals are not allowed here
    }
}

fn main() { }
