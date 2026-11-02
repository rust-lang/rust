fn set_alert(_: bool) {}
fn current_limit() -> i32 {
    100
}

fn main() {
    let used = 120;
    let warning = 90;
    set_alert(used < warning, used > current_limit());
    //~^ ERROR: this function takes 1 argument but 2 arguments were supplied
}
