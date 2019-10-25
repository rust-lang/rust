// Type ascription is unstable

#[cfg(FALSE)]
fn foo() {
    let a = 10: u8; //~ ERROR type ascription is experimental
}

fn main() {}
