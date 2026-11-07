#[cfg(false)]
fn syntax() {
    bar::<Item =   >(); //~ ERROR missing type to the right of `=`
}

fn main() {}
