// run-pass

macro_rules! higher_order {
    (subst $lhs:tt => $rhs:tt) => ({
            macro_rules! anon { $lhs => $rhs }
            anon!(1_usize, 2_usize, "foo")
    });
}

macro_rules! outer {
    ($x:expr; $fragment:ident) => {
        macro_rules! inner { ($y:$fragment) => { $x + $y } }
    }
}

fn main() {
    let val = higher_order!(subst ($x:expr, $y:expr, $foo:expr) => (($x + $y, $foo)));
    assert_eq!(val, (3, "foo"));

    outer!(2; expr);
    assert_eq!(inner!(3), 5);
}
