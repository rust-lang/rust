#[warn(clippy::string_slice)]
#[allow(clippy::no_effect)]

fn main() {
    &"Ölkanne"[1..];
    let m = "Mötörhead";
    &m[2..5];
    let s = String::from(m);
    &s[0..2];
}
