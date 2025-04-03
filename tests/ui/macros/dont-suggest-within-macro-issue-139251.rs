#[macro_export]
macro_rules! is_equal {
    ($left:expr, $right:expr) => {
        $left == $right //~ ERROR can't compare `&{integer}` with `{integer}` [E0277]
    };
}

fn main() {
    let x = 1;
    let y = &x;
    assert!(y == 1); //~ ERROR can't compare `&{integer}` with `{integer}` [E0277]
    assert_eq!(y, 2); //~ ERROR can't compare `&{integer}` with `{integer}` [E0277]
    is_equal!(y, 2);
}
