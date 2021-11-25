fn sum<I: Iterator<Item = ()>>(i: I) -> i32 where I::Item = i32 {
//~^ ERROR equality constraints are not yet supported in `where` clauses
    panic!()
}
fn sum2<I: Iterator>(i: I) -> i32 where I::Item = i32 {
//~^ ERROR equality constraints are not yet supported in `where` clauses
    panic!()
}
fn sum3<J: Iterator>(i: J) -> i32 where I::Item = i32 {
//~^ ERROR equality constraints are not yet supported in `where` clauses
//~| ERROR failed to resolve: use of undeclared type `I`
    panic!()
}

fn main() {}
