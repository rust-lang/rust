//!
#![allow(dead_code)]

/// Test that a database with a key/value that is not `Send` will,
/// indeed, not be `Send`.
///
/// ```compile_fail,E0277
/// use std::rc::Rc;
///
/// #[salsa::query_group(NoSendSyncStorage)]
/// trait NoSendSyncDatabase: salsa::Database {
///     fn no_send_sync_value(&self, key: bool) -> Rc<bool>;
///     fn no_send_sync_key(&self, key: Rc<bool>) -> bool;
/// }
///
/// fn no_send_sync_value(_db: &dyn NoSendSyncDatabase, key: bool) -> Rc<bool> {
///     Rc::new(key)
/// }
///
/// fn no_send_sync_key(_db: &dyn NoSendSyncDatabase, key: Rc<bool>) -> bool {
///     *key
/// }
///
/// #[salsa::database(NoSendSyncStorage)]
/// #[derive(Default)]
/// struct DatabaseImpl {
///     storage: salsa::Storage<Self>,
/// }
///
/// impl salsa::Database for DatabaseImpl {
/// }
///
/// fn is_send<T: Send>(_: T) { }
///
/// fn assert_send() {
///    is_send(DatabaseImpl::default());
/// }
/// ```
fn test_key_not_send_db_not_send() {}

/// Test that a database with a key/value that is not `Sync` will not
/// be `Send`.
///
/// ```compile_fail,E0277
/// use std::rc::Rc;
/// use std::cell::Cell;
///
/// #[salsa::query_group(NoSendSyncStorage)]
/// trait NoSendSyncDatabase: salsa::Database {
///     fn no_send_sync_value(&self, key: bool) -> Cell<bool>;
///     fn no_send_sync_key(&self, key: Cell<bool>) -> bool;
/// }
///
/// fn no_send_sync_value(_db: &dyn NoSendSyncDatabase, key: bool) -> Cell<bool> {
///     Cell::new(key)
/// }
///
/// fn no_send_sync_key(_db: &dyn NoSendSyncDatabase, key: Cell<bool>) -> bool {
///     *key
/// }
///
/// #[salsa::database(NoSendSyncStorage)]
/// #[derive(Default)]
/// struct DatabaseImpl {
///     runtime: salsa::Storage<Self>,
/// }
///
/// impl salsa::Database for DatabaseImpl {
/// }
///
/// fn is_send<T: Send>(_: T) { }
///
/// fn assert_send() {
///    is_send(DatabaseImpl::default());
/// }
/// ```
fn test_key_not_sync_db_not_send() {}

/// Test that a database with a key/value that is not `Sync` will
/// not be `Sync`.
///
/// ```compile_fail,E0277
/// use std::cell::Cell;
/// use std::rc::Rc;
///
/// #[salsa::query_group(NoSendSyncStorage)]
/// trait NoSendSyncDatabase: salsa::Database {
///     fn no_send_sync_value(&self, key: bool) -> Cell<bool>;
///     fn no_send_sync_key(&self, key: Cell<bool>) -> bool;
/// }
///
/// fn no_send_sync_value(_db: &dyn NoSendSyncDatabase, key: bool) -> Cell<bool> {
///     Cell::new(key)
/// }
///
/// fn no_send_sync_key(_db: &dyn NoSendSyncDatabase, key: Cell<bool>) -> bool {
///     *key
/// }
///
/// #[salsa::database(NoSendSyncStorage)]
/// #[derive(Default)]
/// struct DatabaseImpl {
///     runtime: salsa::Storage<Self>,
/// }
///
/// impl salsa::Database for DatabaseImpl {
/// }
///
/// fn is_sync<T: Sync>(_: T) { }
///
/// fn assert_send() {
///    is_sync(DatabaseImpl::default());
/// }
/// ```
fn test_key_not_sync_db_not_sync() {}
