// Issue #922
fn f2(-thing: fn@()) { }

fn f(-thing: fn@()) {
    f2(move thing);
}

fn main() {
    f(fn@() {});
}