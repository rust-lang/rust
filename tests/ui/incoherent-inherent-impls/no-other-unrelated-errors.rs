// E0116 caused other unrelated errors, so check no unrelated errors are emitted.

fn main() {
    let x = "hello";
    x.split(" ");
}

impl Vec<usize> {} //~ ERROR E0116
