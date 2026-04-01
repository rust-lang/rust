#![warn(clippy::manual_strip)]
//@no-rustfix
fn main() {
    let s = "abc";

    if s.starts_with("ab") {
        str::to_string(&s["ab".len()..]);
        //~^ manual_strip

        s["ab".len()..].to_string();

        str::to_string(&s[2..]);
        s[2..].to_string();
    }

    if s.ends_with("bc") {
        str::to_string(&s[..s.len() - "bc".len()]);
        //~^ manual_strip

        s[..s.len() - "bc".len()].to_string();

        str::to_string(&s[..s.len() - 2]);
        s[..s.len() - 2].to_string();
    }

    // Character patterns
    if s.starts_with('a') {
        str::to_string(&s[1..]);
        //~^ manual_strip

        s[1..].to_string();
    }

    // Variable prefix
    let prefix = "ab";
    if s.starts_with(prefix) {
        str::to_string(&s[prefix.len()..]);
        //~^ manual_strip
    }

    // Constant prefix
    const PREFIX: &str = "ab";
    if s.starts_with(PREFIX) {
        str::to_string(&s[PREFIX.len()..]);
        //~^ manual_strip

        str::to_string(&s[2..]);
    }

    // Constant target
    const TARGET: &str = "abc";
    if TARGET.starts_with(prefix) {
        str::to_string(&TARGET[prefix.len()..]);
        //~^ manual_strip
    }

    // String target - not mutated.
    let s1: String = "abc".into();
    if s1.starts_with("ab") {
        s1[2..].to_uppercase();
        //~^ manual_strip
    }

    // String target - mutated. (Don't lint.)
    let mut s2: String = "abc".into();
    if s2.starts_with("ab") {
        s2.push('d');
        s2[2..].to_uppercase();
    }

    // Target not stripped. (Don't lint.)
    let s3 = String::from("abcd");
    let s4 = String::from("efgh");
    if s3.starts_with("ab") {
        s4[2..].to_string();
    }

    // Don't propose to reuse the `stripped` identifier as it is overridden
    if s.starts_with("ab") {
        let stripped = &s["ab".len()..];
        //~^ ERROR: stripping a prefix manually
        let stripped = format!("{stripped}-");
        println!("{stripped}{}", &s["ab".len()..]);
    }

    // Don't propose to reuse the `stripped` identifier as it is mutable
    if s.starts_with("ab") {
        let mut stripped = &s["ab".len()..];
        //~^ ERROR: stripping a prefix manually
        stripped = "";
        let stripped = format!("{stripped}-");
        println!("{stripped}{}", &s["ab".len()..]);
    }
}

#[clippy::msrv = "1.44"]
fn msrv_1_44() {
    let s = "abc";
    if s.starts_with('a') {
        s[1..].to_string();
    }
}

#[clippy::msrv = "1.45"]
fn msrv_1_45() {
    let s = "abc";
    if s.starts_with('a') {
        s[1..].to_string();
        //~^ manual_strip
    }
}
