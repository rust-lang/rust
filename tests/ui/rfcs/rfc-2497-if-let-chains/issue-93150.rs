#[rustfmt::skip]
fn main() {
    match true {
        _ if let true = true && true => {}
        //~^ ERROR `let` expressions in this position are unstable
        _ => {}
    }
}
