fn bare() {}

fn likes_block(f: block()) { f() }

fn main() {
    likes_block(bare);
}