#![crate_name = "foo"]

//@ has foo/index.html '//dd' 'fooo'
//@ !has foo/index.html '//dd//h1' 'fooo'

//@ has foo/fn.foo.html '//h2[@id="fooo"]' 'fooo'
//@ has foo/fn.foo.html '//h2[@id="fooo"]/a[@href="#fooo"]' '§'
/// # fooo
///
/// foo
pub fn foo() {}

//@ has foo/index.html '//dd' 'mooood'
//@ !has foo/index.html '//dd//h2' 'mooood'

//@ has foo/foo/index.html '//h3[@id="mooood"]' 'mooood'
//@ has foo/foo/index.html '//h3[@id="mooood"]/a[@href="#mooood"]' '§'
/// ## mooood
///
/// foo mod
pub mod foo {}

//@ has foo/index.html '//dd/a[@href="https://nougat.world"]/code' 'nougat'

/// [`nougat`](https://nougat.world)
pub struct Bar;
