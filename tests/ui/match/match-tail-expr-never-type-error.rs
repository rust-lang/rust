fn never() -> ! {
    loop {}
}

fn bar(a: bool) {
    match a {
        true => 1,
        false => {
            never() //~ ERROR `match` arms have incompatible types
        }
    }
}
fn main() {
    bar(true);
    bar(false);
}
