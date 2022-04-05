// Test that we correctly consider the type of `match` to be the LUB
// of the various arms, particularly in the case where regions are
// involved.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

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
        s
        //[base]~^ ERROR E0312
        //[nll]~^^ ERROR lifetime may not live long enough
    }
}

pub fn opt_str3<'a>(maybestr: &'a Option<String>) -> &'static str {
    if maybestr.is_some() {
        let s: &'a str = maybestr.as_ref().unwrap();
        s
        //[base]~^ ERROR E0312
        //[nll]~^^ ERROR lifetime may not live long enough
    } else {
        "(none)"
    }
}


fn main() {}
