#[warn(clippy::string_slice)]
#[allow(clippy::no_effect)]

fn main() {
    &"Ölkanne"[1..];
    //~^ ERROR: indexing into a string may panic if the index is within a UTF-8 character
    //~| NOTE: `-D clippy::string-slice` implied by `-D warnings`
    let m = "Mötörhead";
    &m[2..5];
    //~^ ERROR: indexing into a string may panic if the index is within a UTF-8 character
    let s = String::from(m);
    &s[0..2];
    //~^ ERROR: indexing into a string may panic if the index is within a UTF-8 character
}
