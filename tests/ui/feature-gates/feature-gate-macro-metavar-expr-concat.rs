macro_rules! join {
    ($lhs:ident, $rhs:ident) => {
        let ${concat($lhs, $rhs)}: () = ();
        //~^ ERROR the `concat` meta-variable expression is unstable
    };
}

fn main() {
}
