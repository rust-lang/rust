macro_rules! mac {
    ( $($v:tt)* ) => (
        $v  //~ ERROR still repeating at this depth
    )
}

fn main() {
    mac!(0);
}
