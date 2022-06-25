macro_rules! x {
    ($($c:tt)*) => {
        $($c)ö* //~ ERROR macro expansion ends with an incomplete expression: expected expression
    };
}

fn main() {
    x!(!);
}
