#[cfg(FALSE)]
fn if_else_parse_error() {
    if true {
    } #[attr] else if false { //~ ERROR expected
    }
}

#[cfg(FALSE)]
fn else_attr_ifparse_error() {
    if true {
    } else #[attr] if false { //~ ERROR outer attributes are not allowed
    } else {
    }
}

#[cfg(FALSE)]
fn else_parse_error() {
    if true {
    } else if false {
    } #[attr] else { //~ ERROR expected
    }
}

fn main() {
}
