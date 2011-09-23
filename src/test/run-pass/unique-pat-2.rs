type foo = {a: int, b: uint};
tag bar { u(~foo); w(int); }

fn main() {
    assert (alt u(~{a: 10, b: 40u}) {
              u(~{a: a, b: b}) { a + (b as int) }
              _ { 66 }
            } == 50);
}
