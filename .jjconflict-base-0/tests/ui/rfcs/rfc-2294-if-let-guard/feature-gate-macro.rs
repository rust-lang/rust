// gate-test-if_let_guard

fn main() {
    macro_rules! use_expr {
        ($e:expr) => {
            match () {
                () if $e => {}
                _ => {}
            }
        }
    }
    use_expr!(let 0 = 1);
    //~^ ERROR no rules expected keyword `let`
}
