//@ check-pass

#[cfg(false)]
fn simple_attr() {
    #[attr] if true {}
    #[allow_warnings] if true {}
}

#[cfg(false)]
fn if_else_chain() {
    #[first_attr] if true {
    } else if false {
    } else {
    }
}

#[cfg(false)]
fn if_let() {
    #[attr] if let Some(_) = Some(true) {}
}

fn bar() {
    #[cfg(false)]
    if true {
        let x: () = true; // Should not error due to the #[cfg(false)]
    }

    #[cfg_attr(not(FALSE), cfg(false))]
    if true {
        let a: () = true; // Should not error due to the applied #[cfg(false)]
    }
}

macro_rules! custom_macro {
    ($expr:expr) => {}
}

custom_macro! {
    #[attr] if true {}
}


fn main() {}
