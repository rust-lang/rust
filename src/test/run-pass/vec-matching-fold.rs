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

fn foldr<T, U: Copy>(
    values: &[T],
    initial: U,
    function: &fn(element: &T, partial: U) -> U
) -> U {
    match values {
        [..head, tail] =>
            foldr(head, function(&tail, initial), function),
        _ => copy initial
    }
}

fn main() {
    let x = [1, 2, 3, 4, 5];

    let product = foldl(x, 1, |a, b| a * *b);
    assert product == 120;

    let sum = foldr(x, 0, |a, b| *a + b);
    assert sum == 15;
}
