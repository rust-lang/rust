#[cfg(FALSE)]
pub fn foo() {
    'a: { //~ ERROR labels on blocks are unstable
        break 'a;
    }
}

fn main() {}
