// Check that when there are vacuous predicates in the environment
// (which make a fn uncallable) we don't erroneously cache those and
// then consider them satisfied elsewhere. The current technique for
// doing this is to not use global caches when there is a chance that
// the environment contains such a predicate.
// We still error for `i32: Bar<u32>` pending #48214

trait Foo<X,Y>: Bar<X> {
}

trait Bar<X> { }

// We don't always check where clauses for sanity, but in this case
// wfcheck does report an error here:
fn vacuous<A>()
    where i32: Foo<u32, A> //~ ERROR the trait bound `i32: Bar<u32>` is not satisfied
{
    // ... the original intention was to check that we don't use that
    // vacuous where clause (which could never be satisfied) to accept
    // the following line and then mess up calls elsewhere.
    require::<i32, u32>();
}

fn require<A,B>()
    where A: Bar<B>
{
}

fn main() {
    require::<i32, u32>();
}
