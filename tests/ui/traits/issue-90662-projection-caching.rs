// check-pass

// Regression test for issue #90662
// Tests that projection caching does not cause a spurious error

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
