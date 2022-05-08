// Test that we correctly consider the type of `match` to be the LUB
// of the various arms, particularly in the case where regions are
// involved.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

pub fn opt_str0<'a>(maybestr: &'a Option<String>) -> &'a str {
    match *maybestr {
        Some(ref s) => {
            let s: &'a str = s;
            s
        }
        None => "(none)",
    }
}

pub fn opt_str1<'a>(maybestr: &'a Option<String>) -> &'a str {
    match *maybestr {
        None => "(none)",
        Some(ref s) => {
            let s: &'a str = s;
            s
        }
    }
}

pub fn opt_str2<'a>(maybestr: &'a Option<String>) -> &'static str {
    match *maybestr {
        None => "(none)",
        Some(ref s) => {
            let s: &'a str = s;
            s
            //[base]~^ ERROR E0312
            //[nll]~^^ ERROR lifetime may not live long enough
        }
    }
}

pub fn opt_str3<'a>(maybestr: &'a Option<String>) -> &'static str {
    match *maybestr {
        Some(ref s) => {
            let s: &'a str = s;
            s
            //[base]~^ ERROR E0312
            //[nll]~^^ ERROR lifetime may not live long enough
        }
        None => "(none)",
    }
}

fn main() {}
