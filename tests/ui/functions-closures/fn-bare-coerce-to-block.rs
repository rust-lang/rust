//@ run-pass

fn bare() {}

fn likes_block<F>(f: F) where F: FnOnce() { f() }

pub fn main() {
    likes_block(bare);
}
