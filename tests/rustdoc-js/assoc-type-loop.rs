#![crate_name = "foo"]

// reduced from sqlx 0.7.3
use std::future::Future;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
pub enum Error {}
pub trait Acquire<'c> {
    type Database: Database;
    type Connection: Deref<Target = <Self::Database as Database>::Connection> + DerefMut + Send;
}
pub trait Database {
    type Connection: Connection<Database = Self>;
}
pub trait Connection {
    type Database: Database;
    type Options: ConnectionOptions<Connection = Self>;
    fn begin(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<Transaction<'_, Self::Database>, Error>> + Send + '_>>
    where
        Self: Sized;
}
pub trait ConnectionOptions {
    type Connection: Connection;
}
pub struct Transaction<'c, DB: Database> {
    _db: &'c DB,
}
impl<'t, 'c, DB: Database> Acquire<'t> for &'t mut Transaction<'c, DB>
where
    <DB as Database>::Connection: Send,
{
    type Database = DB;
    type Connection = &'t mut <DB as Database>::Connection;
}
