// Regression test for https://github.com/rust-lang/rust/issues/100605

fn takes_option(_arg: Option<&String>) {}

fn main() {
    takes_option(&None); //~ ERROR mismatched types [E0308]

    let x = String::from("x");
    let res = Some(x);
    takes_option(&res); //~ ERROR mismatched types [E0308]
}
