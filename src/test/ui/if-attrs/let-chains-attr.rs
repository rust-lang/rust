// check-pass

#![feature(let_chains)] //~ WARN the feature `let_chains` is incomplete

#[cfg(FALSE)]
fn foo() {
    #[attr]
    if let Some(_) = Some(true) && let Ok(_) = Ok(1) {
    } else if let Some(false) = Some(true) {
    }
}

fn main() {}
