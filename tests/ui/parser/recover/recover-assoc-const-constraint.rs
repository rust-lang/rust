#[cfg(false)]
fn syntax() {
    bar::<Item = 42>();
    //~^ ERROR associated const equality is incomplete
    bar::<Item = { 42 }>();
    //~^ ERROR associated const equality is incomplete
}

fn main() {}
