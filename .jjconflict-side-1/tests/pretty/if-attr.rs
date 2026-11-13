//@ pp-exact

#[cfg(false)]
fn simple_attr() {

    #[attr]
    if true {}

    #[allow_warnings]
    if true {}
}

#[cfg(false)]
fn if_else_chain() {

    #[first_attr]
    if true {} else if false {} else {}
}

#[cfg(false)]
fn if_let() {

    #[attr]
    if let Some(_) = Some(true) {}
}

#[cfg(false)]
fn let_attr_if() {
    let _ = #[attr] if let _ = 0 {};
    let _ = #[attr] if true {};

    let _ = #[attr] if let _ = 0 {} else {};
    let _ = #[attr] if true {} else {};
}


fn main() {}
