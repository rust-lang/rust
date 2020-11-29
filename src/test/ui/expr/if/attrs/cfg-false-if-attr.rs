// check-pass

#[cfg(FALSE)]
fn simple_attr() {
    #[attr] if true {}
    #[allow_warnings] if true {}
}

#[cfg(FALSE)]
fn if_else_chain() {
    #[first_attr] if true {
    } else if false {
    } else {
    }
}

#[cfg(FALSE)]
fn if_let() {
    #[attr] if let Some(_) = Some(true) {}
}

fn bar() {
    #[cfg(FALSE)]
    if true {
        let x: () = true; // Should not error due to the #[cfg(FALSE)]
    }

    #[cfg_attr(not(unset_attr), cfg(FALSE))]
    if true {
        let a: () = true; // Should not error due to the applied #[cfg(FALSE)]
    }
}

macro_rules! custom_macro {
    ($expr:expr) => {}
}

custom_macro! {
    #[attr] if true {}
}


fn main() {}
