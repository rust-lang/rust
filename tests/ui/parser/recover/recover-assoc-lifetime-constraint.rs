#[cfg(FALSE)]
fn syntax() {
    bar::<Item = 'a>(); //~ ERROR lifetimes are not permitted in this context
}

fn main() {}
