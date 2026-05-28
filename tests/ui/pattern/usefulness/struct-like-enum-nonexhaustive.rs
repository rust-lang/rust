enum A {
    B { x: Option<isize> },
    C
}

fn main() {
    let x = A::B { x: Some(3) };
    match x {   //~ ERROR non-exhaustive patterns
        A::C => {}
        A::B { x: None } => {}
    }
}
