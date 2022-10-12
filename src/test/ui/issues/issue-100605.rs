fn takes_option(_arg: Option<&String>) {}

fn main() {
    takes_option(&None); //~ ERROR 4:18: 4:23: mismatched types [E0308]

    let x = String::from("x");
    let res = Some(x);
    takes_option(&res); //~ ERROR 8:18: 8:22: mismatched types [E0308]
}
