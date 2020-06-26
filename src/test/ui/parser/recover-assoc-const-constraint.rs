#[cfg(FALSE)]
fn syntax() {
    bar::<Item = 42>(); //~ ERROR cannot constrain an associated constant to a value
    bar::<Item = { 42 }>(); //~ ERROR cannot constrain an associated constant to a value
}

fn main() {}
