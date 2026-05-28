#![deny(invalid_doc_attributes)]
//~^ NOTE

#[doc(include = "external-cross-doc.md")]
//~^ ERROR unknown `doc` attribute `include`
//~| HELP use `doc = include_str!` instead
// FIXME(#85497): make this a deny instead so it's more clear what's happening
pub struct NeedMoreDocs;
