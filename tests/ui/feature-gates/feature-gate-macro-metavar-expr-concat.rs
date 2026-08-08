macro_rules! join {
    ($lhs:ident, $rhs:ident) => {
        let ${concat($lhs, $rhs)}: &'static str = ${concat_str($lhs, $rhs)};
        //~^ ERROR the `concat` meta-variable expression is unstable
        //~| ERROR the `concat_str` meta-variable expression is unstable
    };
}

fn main() {
}
