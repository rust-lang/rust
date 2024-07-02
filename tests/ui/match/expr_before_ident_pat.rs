macro_rules! funny {
    ($a:expr, $b:ident) => {
        match [1, 2] {
            [$a, $b] => {}
        }
    };
}

fn main() {
    funny!(a, a);
    //~^ ERROR cannot find value `a`
    //~| ERROR arbitrary expressions aren't allowed in patterns
}
