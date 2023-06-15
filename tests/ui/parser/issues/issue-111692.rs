mod module {
    #[derive(Eq, PartialEq)]
    pub struct Type {
        pub x: u8,
        pub y: u8,
    }

    pub const C: u8 = 32u8;
}

fn test(x: module::Type) {
    if x == module::Type { x: module::C, y: 1 } { //~ ERROR invalid struct literal
    }
}

fn test2(x: module::Type) {
    if x ==module::Type { x: module::C, y: 1 } { //~ ERROR invalid struct literal
    }
}


fn test3(x: module::Type) {
    if x == Type { x: module::C, y: 1 } { //~ ERROR invalid struct literal
    }
}

fn test4(x: module::Type) {
    if x == demo_module::Type { x: module::C, y: 1 } { //~ ERROR invalid struct literal
    }
}

fn main() { }
