pub fn main() {
    if cfg!(not()) { } //~ ERROR E0536
}
