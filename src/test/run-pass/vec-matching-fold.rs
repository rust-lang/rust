fn foldl<T,U:Clone>(values: &[T],
                    initial: U,
                    function: &fn(partial: U, element: &T) -> U)
                    -> U {
    match values {
        [ref head, ..tail] =>
            foldl(tail, function(initial, head), function),
        [] => initial.clone()
    }
}

fn foldr<T,U:Clone>(values: &[T],
                    initial: U,
                    function: &fn(element: &T, partial: U) -> U)
                    -> U {
    match values {
        [..head, ref tail] =>
            foldr(head, function(tail, initial), function),
        [] => initial.clone()
    }
}

pub fn main() {
    let x = [1, 2, 3, 4, 5];

    let product = foldl(x, 1, |a, b| a * *b);
    assert_eq!(product, 120);

    let sum = foldr(x, 0, |a, b| *a + b);
    assert_eq!(sum, 15);
}
