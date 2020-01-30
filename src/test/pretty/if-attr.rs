// pp-exact

#[cfg(FALSE)]
fn simple_attr() {

    #[attr]
    if true { }

    #[allow_warnings]
    if true { }
}

#[cfg(FALSE)]
fn if_else_chain() {

    #[first_attr]
    if true { } else if false { } else { }
}

#[cfg(FALSE)]
fn if_let() {

    #[attr]
    if let Some(_) = Some(true) { }
}


fn main() { }
