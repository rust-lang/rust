#![feature(type_alias_impl_trait)]

//@ edition:2021

use std::future::Future;

struct Connection {
}

trait Transaction {
}

struct TestTransaction<'conn> {
    conn: &'conn Connection
}

impl<'conn> Transaction for TestTransaction<'conn> {
}

struct Context {
}

type TransactionResult<O> = Result<O, ()>;

type TransactionFuture<'__, O> = impl '__ + Future<Output = TransactionResult<O>>;
//~^ ERROR unconstrained opaque type

fn execute_transaction_fut<'f, F, O>(
    f: F,
) -> impl FnOnce(&mut dyn Transaction) -> TransactionFuture<'_, O>
where
    F: FnOnce(&mut dyn Transaction) -> TransactionFuture<'_, O> + 'f
{
    f
    //~^ ERROR expected generic lifetime parameter, found `'_`
}

impl Context {
    async fn do_transaction<O>(
        &self, f: impl FnOnce(&mut dyn Transaction) -> TransactionFuture<'_, O>
    ) -> TransactionResult<O>
    {
        //~^ ERROR expected generic lifetime parameter, found `'_`
        let mut conn = Connection {};
        let mut transaction = TestTransaction { conn: &mut conn };
        f(&mut transaction).await
    }
}

fn main() {}
