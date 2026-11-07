#[cfg(false)]
fn if_else_parse_error() {
    if true {
    } #[attr] else if false { //~ ERROR expected
    }
}

#[cfg(false)]
fn else_attr_ifparse_error() {
    if true {
    } else #[attr] if false { //~ ERROR outer attributes are not allowed
    } else {
    }
}

#[cfg(false)]
fn else_parse_error() {
    if true {
    } else if false {
    } #[attr] else { //~ ERROR expected
    }
}

fn main() {
}
