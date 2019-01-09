// Test that we correctly consider the type of `match` to be the LUB
// of the various arms, particularly in the case where regions are
// involved.

pub fn opt_str0<'a>(maybestr: &'a Option<String>) -> &'a str {
    if maybestr.is_none() {
        "(none)"
    } else {
        let s: &'a str = maybestr.as_ref().unwrap();
        s
    }
}

pub fn opt_str1<'a>(maybestr: &'a Option<String>) -> &'a str {
    if maybestr.is_some() {
        let s: &'a str = maybestr.as_ref().unwrap();
        s
    } else {
        "(none)"
    }
}

pub fn opt_str2<'a>(maybestr: &'a Option<String>) -> &'static str {
    if maybestr.is_none() {
        "(none)"
    } else {
        let s: &'a str = maybestr.as_ref().unwrap();
        s  //~ ERROR E0312
    }
}

pub fn opt_str3<'a>(maybestr: &'a Option<String>) -> &'static str {
    if maybestr.is_some() {
        let s: &'a str = maybestr.as_ref().unwrap();
        s  //~ ERROR E0312
    } else {
        "(none)"
    }
}


fn main() {}
