macro_rules! x {
    ($($c:tt)*) => {
        $($c)ö*
    };
}

fn main() {
    x!(!); //~ ERROR macro expansion ends with an incomplete expression: expected expression
}
