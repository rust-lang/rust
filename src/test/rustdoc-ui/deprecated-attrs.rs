// check-pass
// compile-flags: --passes unknown-pass
// error-pattern: the `passes` flag no longer functions

#![doc(no_default_passes)]
//~^ WARNING attribute is deprecated
//~| NOTE see issue #44136
//~| HELP no longer functions; you may want to use `#![doc(document_private_items)]`
#![doc(passes = "collapse-docs unindent-comments")]
//~^ WARNING attribute is deprecated
//~| NOTE see issue #44136
//~| HELP no longer functions; you may want to use `#![doc(document_private_items)]`
#![doc(plugins = "xxx")]
//~^ WARNING attribute is deprecated
//~| NOTE see issue #44136
//~| WARNING no longer functions; see CVE
