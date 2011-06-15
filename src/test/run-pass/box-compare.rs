

fn main() {
    assert (@1 < @3);
    assert (@@"hello " > @@"hello");
    assert (@@@"hello" != @@@"there");
}