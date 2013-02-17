pub fn main() {
    let x = [1, 2, 3, 4, 5];
    match x {
        [a, b, c, d, e, f] => {
            ::core::util::unreachable();
        }
        [a, b, c, d, e] => {
            assert a == 1;
            assert b == 2;
            assert c == 3;
            assert d == 4;
            assert e == 5;
        }
        _ => {
            ::core::util::unreachable();
        }
    }
}
