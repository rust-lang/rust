// run-rustfix

// These are forbidden occurrences of label-break-value

#[allow(unused_unsafe)]
fn labeled_unsafe() {
    unsafe 'b: {} //~ ERROR block label not supported here
}

fn labeled_if() {
    if true 'b: {} //~ ERROR block label not supported here
}

fn labeled_else() {
    if true {} else 'b: {} //~ ERROR block label not supported here
}

fn labeled_match() {
    match false 'b: { //~ ERROR block label not supported here
        _ => {}
    }
}

fn main() {
    labeled_unsafe();
    labeled_if();
    labeled_else();
    labeled_match();
}
