#[cfg(FALSE)]
fn syntax() {
    bar::<Item = 'a>(); //~ ERROR associated lifetimes are not supported
}

fn main() {}
