enum A {
    B { x: Option<isize> },
    C
}

fn main() {
    let x = A::B { x: Some(3) };
    match x {   //~ ERROR match is non-exhaustive
        A::C => {}
        A::B { x: None } => {}
    }
}
