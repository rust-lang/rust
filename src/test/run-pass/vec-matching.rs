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

pub fn main() {
    let x = [1, 2, 3, 4, 5];
    match x {
        [a, b, c, d, e, f] => {
            ::core::util::unreachable();
        }
        [a, b, c, d, e] => {
            fail_unless!(a == 1);
            fail_unless!(b == 2);
            fail_unless!(c == 3);
            fail_unless!(d == 4);
            fail_unless!(e == 5);
        }
        _ => {
            ::core::util::unreachable();
        }
    }

    let product = foldl(x, 1, |a, b| a * *b);
    fail_unless!(product == 120);
}
