struct S {
    //~^ ERROR E0072
    element: Option<S>
}

fn main() {
    let x = S { element: None };
}
