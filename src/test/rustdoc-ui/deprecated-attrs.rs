// check-pass
// compile-flags: --passes unknown-pass
// error-pattern: ignoring unknown pass `unknown-pass`

#![doc(no_default_passes)]
//~^ WARNING attribute is deprecated
//~| NOTE see issue #44136
//~| HELP use `#![doc(document_private_items)]`
#![doc(passes = "collapse-docs unindent-comments")]
//~^ WARNING attribute is deprecated
//~| NOTE see issue #44136
//~| WARNING ignoring unknown pass
//~| NOTE `collapse-docs` pass was removed
#![doc(plugins = "xxx")]
//~^ WARNING attribute is deprecated
//~| NOTE see issue #44136
//~| WARNING no longer functions; see CVE
