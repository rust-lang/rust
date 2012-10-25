enum A {
    B { x: Option<int> },
    C
}

fn main() {
    let x = B { x: Some(3) };
    match x {   //~ ERROR non-exhaustive patterns
        C => {}
        B { x: None } => {}
    }
}


