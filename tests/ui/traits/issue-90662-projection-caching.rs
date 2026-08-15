//@ revisions: old next
//@[next] compile-flags: -Znext-solver=coherence
//@ check-pass

// Regression test for issue #90662
// Tests that projection caching does not cause a spurious error.
// Coherence relies on the following overflowing goal to still constrain
// `?0` to `dyn Service`.
//
//     Projection(<ServiceImpl as Provider<TestModule>>::Interface. ?0)
//
// cc https://github.com/rust-lang/trait-system-refactor-initiative/issues/70.

trait HasProvider<T: ?Sized> {}
trait Provider<M> {
    type Interface: ?Sized;
}

trait Repository {}
trait Service {}

struct DbConnection;
impl<M> Provider<M> for DbConnection {
    type Interface = DbConnection;
}

struct RepositoryImpl;
impl<M: HasProvider<DbConnection>> Provider<M> for RepositoryImpl {
    type Interface = dyn Repository;
}

struct ServiceImpl;
impl<M: HasProvider<dyn Repository>> Provider<M> for ServiceImpl {
    type Interface = dyn Service;
}

struct TestModule;
impl HasProvider<<DbConnection as Provider<Self>>::Interface> for TestModule {}
impl HasProvider<<RepositoryImpl as Provider<Self>>::Interface> for TestModule {}
impl HasProvider<<ServiceImpl as Provider<Self>>::Interface> for TestModule {}

fn main() {}
