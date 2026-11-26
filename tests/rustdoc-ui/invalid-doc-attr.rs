// Ensures that invalid form of (some) doc attributes doesn't go unnoticed.

#![deny(invalid_doc_attributes)]

#[doc(hidden = "for some reason")]
//~^ ERROR didn't expect any arguments here
//~| WARN this was previously accepted by the compiler
#[doc(no_inline = "tada")]
//~^ ERROR didn't expect any arguments here
//~| WARN this was previously accepted by the compiler
pub fn foo() {}

#[doc(hidden("for some reason"))]
//~^ ERROR didn't expect any arguments here
//~| WARN this was previously accepted by the compiler
#[doc(no_inline("tada"))]
//~^ ERROR didn't expect any arguments here
//~| WARN this was previously accepted by the compiler
pub fn foo2() {}
