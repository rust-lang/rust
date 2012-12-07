fn foldl<T, U: Copy>(
    values: &[T],
    initial: U,
    function: &fn(partial: U, element: &T) -> U
) -> U {
    match values {
        [head, ..tail] =>
            foldl(tail, function(initial, &head), function),
        _ => copy initial
    }
}

fn main() {
    let x = [1, 2, 3, 4, 5];
    match x {
        [a, b, c, d, e, f] => {
            core::util::unreachable();
        }
        [a, b, c, d, e] => {
            assert a == 1;
            assert b == 2;
            assert c == 3;
            assert d == 4;
            assert e == 5;
        }
        _ => {
            core::util::unreachable();
        }
    }

    let product = foldl(x, 1, |a, b| a * *b);
    assert product == 120;
}
