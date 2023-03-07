// check-pass

#![feature(let_chains)]

#[cfg(FALSE)]
fn foo() {
    #[attr]
    if let Some(_) = Some(true) && let Ok(_) = Ok(1) {
    } else if let Some(false) = Some(true) {
    }
}

fn main() {}
