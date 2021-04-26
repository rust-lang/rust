#![crate_type = "lib"]

#[doc = "test"]
#[doc(codeblock_attr = "1,2,3")] //~ ERROR
mod a {}

#[doc = "test"]
#[doc(codeblock_attr = "foo bar")] //~ ERROR
mod b {}

#[doc = "test"]
#[doc(codeblock_attr(" ", ","))] //~ ERROR
//~^ ERROR
mod c {}

#[doc = "test"]
#[doc(codeblock_attr("rust"))] // OK!
mod d {}

#[doc = "test"]
#[doc(codeblock_attr = "rust")] // OK!
mod e {}
