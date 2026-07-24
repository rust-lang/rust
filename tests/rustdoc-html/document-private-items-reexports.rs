//@ compile-flags: --document-private-items

#![crate_name = "document_private_items_reexports"]

// @has document_private_items_reexports/index.html
// @hasraw - 'ABC'
// @hasraw - 'foo bar'
// @has document_private_items_reexports/struct.ABC.html
// @hasraw - 'pub(crate) struct ABC'

#[doc(hidden)]
pub(crate) struct __ABC;

/// foo bar
#[doc(inline)]
pub(crate) use __ABC as ABC;
