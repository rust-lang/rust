// Fix for <https://github.com/rust-lang/rust/issues/125432>.

fn separate_arms() {
    let mut x = None;
    match x {
        None => {
            x = Some(0);
        }
        Some(right) => {
            consume(right);
            //~^ ERROR cannot find function `consume` in this scope
        }
    }
}

fn main() {}
