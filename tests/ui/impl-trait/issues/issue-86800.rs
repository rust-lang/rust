#![feature(type_alias_impl_trait)]

// edition:2021
// unset-rustc-env:RUST_BACKTRACE
// compile-flags:-Z treat-err-as-bug=1
// error-pattern:stack backtrace:
// failure-status:101
// normalize-stderr-test "note: .*" -> ""
// normalize-stderr-test "thread 'rustc' .*" -> ""
// normalize-stderr-test " +[0-9]+:.*\n" -> ""
// normalize-stderr-test " +at .*\n" -> ""

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

fn execute_transaction_fut<'f, F, O>(
    f: F,
) -> impl FnOnce(&mut dyn Transaction) -> TransactionFuture<'_, O>
where
    F: FnOnce(&mut dyn Transaction) -> TransactionFuture<'_, O> + 'f
{
    f
}

impl Context {
    async fn do_transaction<O>(
        &self, f: impl FnOnce(&mut dyn Transaction) -> TransactionFuture<'_, O>
    ) -> TransactionResult<O>
    {
        let mut conn = Connection {};
        let mut transaction = TestTransaction { conn: &mut conn };
        f(&mut transaction).await
    }
}

fn main() {}
