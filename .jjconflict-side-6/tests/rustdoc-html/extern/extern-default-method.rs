//@ aux-build:rustdoc-extern-default-method.rs
//@ ignore-cross-compile
// ignore-tidy-linelength

extern crate rustdoc_extern_default_method as ext;

// For this test, the dependency is compiled but not documented.
//
// Still, the struct from the external crate and its impl should be documented since
// the struct is re-exported from this crate.
// However, the method in the trait impl should *not* have a link (an `href` attribute) to
// its corresponding item in the trait declaration since it would otherwise be broken.
//
// In older versions of rustdoc, the impl item (`a[@class="fn"]`) used to link to
// `#method.provided` – i.e. "to itself". Put in quotes since that was actually incorrect in
// general: If the type `Struct` also had an inherent method called `provided`, the impl item
// would link to that one even though those two methods are distinct items!

//@ count extern_default_method/struct.Struct.html '//*[@id="method.provided"]' 1
//@ count extern_default_method/struct.Struct.html '//*[@id="method.provided"]//a[@class="fn"]' 1
//@ snapshot no_href_on_anchor - '//*[@id="method.provided"]//a[@class="fn"]'
//@ has extern_default_method/struct.Struct.html '//*[@id="method.provided"]//a[@class="anchor"]/@href' #method.provided
pub use ext::Struct;
