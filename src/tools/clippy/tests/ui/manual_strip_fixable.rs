#![warn(clippy::manual_strip)]

fn main() {
    let s = "abc";

    if s.starts_with("ab") {
        let stripped = &s["ab".len()..];
        //~^ ERROR: stripping a prefix manually
        println!("{stripped}{}", &s["ab".len()..]);
    }

    if s.ends_with("bc") {
        let stripped = &s[..s.len() - "bc".len()];
        //~^ ERROR: stripping a suffix manually
        println!("{stripped}{}", &s[..s.len() - "bc".len()]);
    }
}
