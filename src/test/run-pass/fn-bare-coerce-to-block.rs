fn bare() {}

fn likes_block(f: fn()) { f() }

fn main() {
    likes_block(bare);
}