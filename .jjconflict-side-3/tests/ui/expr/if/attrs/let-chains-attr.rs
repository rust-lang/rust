//@ check-pass
//@ edition:2024

#[cfg(false)]
fn foo() {
    #[attr]
    if let Some(_) = Some(true) && let Ok(_) = Ok(1) {
    } else if let Some(false) = Some(true) {
    }
}

fn main() {}
