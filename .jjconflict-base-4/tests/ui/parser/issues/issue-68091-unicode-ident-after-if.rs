macro_rules! x {
    ($($c:tt)*) => {
        $($c)ö* {}
        //~^ ERROR missing condition for `if` expression
    };
}

fn main() {
    x!(if);
}
