// check-pass
// Regression test for #28871. The problem is that rustc encountered
// two ways to project, one from a where clause and one from the where
// clauses on the trait definition. (In fact, in this case, the where
// clauses originated from the trait definition as well.) The true
// cause of the error is that the trait definition where clauses are
// not being normalized, and hence the two sources are considered in
// conflict, and not a duplicate. Hacky solution is to prefer where
// clauses over the data found in the trait definition.

trait T {
    type T;
}

struct S;
impl T for S {
    type T = S;
}

trait T2 {
    type T: Iterator<Item=<S as T>::T>;
}

fn main() { }
