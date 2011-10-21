fn bare() {}

fn likes_shared(f: fn@()) { f() }

fn main() {
    likes_shared(bare);
}