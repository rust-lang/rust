macro_rules! x {
    ($($c:tt)*) => {
        $($c)ö* {} //~ ERROR missing condition for `if` expression
    };             //~| ERROR mismatched types
}

fn main() {
    x!(if);
}
