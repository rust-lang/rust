// Ensures that invalid form of (some) doc attributes doesn't go unnoticed.

#[doc(hidden = "for some reason")]
//~^ ERROR malformed `doc` attribute input
#[doc(no_inline = "tada")]
//~^ ERROR malformed `doc` attribute input
pub fn foo() {}

#[doc(hidden("for some reason"))]
//~^ ERROR malformed `doc` attribute input
#[doc(no_inline("tada"))]
//~^ ERROR malformed `doc` attribute input
pub fn foo2() {}
