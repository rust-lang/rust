#[allow(unreachable_code)]

fn main() {
    loop {
        break while continue { //~ ERROR E0590
        }
    }
}
