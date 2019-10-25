#[cfg(FALSE)]
fn foo() {
    match 22 {
        0 .. 3 => {} //~ ERROR exclusive range pattern syntax is experimental
        PATH .. 3 => {} //~ ERROR exclusive range pattern syntax is experimental
        _ => {}
    }
}

fn main() {}
