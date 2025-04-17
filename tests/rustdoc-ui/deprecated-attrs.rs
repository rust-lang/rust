//@ compile-flags: --passes unknown-pass

#![doc(no_default_passes)]
//~^ ERROR unknown `doc` attribute `no_default_passes`
//~| NOTE no longer functions
//~| NOTE see issue #44136
//~| HELP you may want to use `doc(document_private_items)`
//~| NOTE `doc(no_default_passes)` is now a no-op
//~| NOTE `#[deny(invalid_doc_attributes)]` on by default
#![doc(passes = "collapse-docs unindent-comments")]
//~^ ERROR unknown `doc` attribute `passes`
//~| NOTE no longer functions
//~| NOTE see issue #44136
//~| HELP you may want to use `doc(document_private_items)`
//~| NOTE `doc(passes)` is now a no-op
#![doc(plugins = "xxx")]
//~^ ERROR unknown `doc` attribute `plugins`
//~| NOTE see issue #44136
//~| NOTE no longer functions
//~| NOTE `doc(plugins)` is now a no-op

//~? WARN the `passes` flag no longer functions
//~? NOTE see issue #44136
//~? HELP you may want to use --document-private-items
