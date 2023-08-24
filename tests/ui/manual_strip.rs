#![warn(clippy::manual_strip)]
//@no-rustfix
fn main() {
    let s = "abc";

    if s.starts_with("ab") {
        str::to_string(&s["ab".len()..]);
        //~^ ERROR: stripping a prefix manually
        s["ab".len()..].to_string();

        str::to_string(&s[2..]);
        s[2..].to_string();
    }

    if s.ends_with("bc") {
        str::to_string(&s[..s.len() - "bc".len()]);
        //~^ ERROR: stripping a suffix manually
        s[..s.len() - "bc".len()].to_string();

        str::to_string(&s[..s.len() - 2]);
        s[..s.len() - 2].to_string();
    }

    // Character patterns
    if s.starts_with('a') {
        str::to_string(&s[1..]);
        //~^ ERROR: stripping a prefix manually
        s[1..].to_string();
    }

    // Variable prefix
    let prefix = "ab";
    if s.starts_with(prefix) {
        str::to_string(&s[prefix.len()..]);
        //~^ ERROR: stripping a prefix manually
    }

    // Constant prefix
    const PREFIX: &str = "ab";
    if s.starts_with(PREFIX) {
        str::to_string(&s[PREFIX.len()..]);
        //~^ ERROR: stripping a prefix manually
        str::to_string(&s[2..]);
    }

    // Constant target
    const TARGET: &str = "abc";
    if TARGET.starts_with(prefix) {
        str::to_string(&TARGET[prefix.len()..]);
        //~^ ERROR: stripping a prefix manually
    }

    // String target - not mutated.
    let s1: String = "abc".into();
    if s1.starts_with("ab") {
        s1[2..].to_uppercase();
        //~^ ERROR: stripping a prefix manually
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
        //~^ ERROR: stripping a prefix manually
    }
}
