//@ run-pass
#![allow(path_statements)]
#![allow(unused_variables)]
// Regression test for issue #19499. Due to incorrect caching of trait
// results for closures with upvars whose types were not fully
// computed, this rather bizarre little program (along with many more
// reasonable examples) let to ambiguity errors about not being able
// to infer sufficient type information.


fn main() {
    let n = 0;
    let it = Some(1_usize).into_iter().inspect(|_| {n;});
}
