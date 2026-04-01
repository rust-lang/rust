// Identifier pattern referring to a const generic parameter is an error (issue #68853).

fn check<const N: usize>() {
    match 1 {
        N => {} //~ ERROR constant parameters cannot be referenced in patterns
        _ => {}
    }
}

fn main() {}
