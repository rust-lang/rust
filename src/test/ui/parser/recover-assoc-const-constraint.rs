// run-pass

#[cfg(FALSE)]
fn syntax() {
    bar::<Item = 42>();
    bar::<Item = { 42 }>();
}

fn main() {}
