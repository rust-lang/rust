struct S {
    //~^ ERROR E0072
    //~| ERROR cycle
    element: Option<S>
}

fn main() {
    let x = S { element: None };
}
