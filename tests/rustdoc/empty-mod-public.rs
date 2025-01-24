//@ has 'empty_mod_public/index.html' '//a[@href="foo/index.html"]' 'foo'
//@ hasraw 'empty_mod_public/sidebar-items.js' 'foo'
//@ matches 'empty_mod_public/foo/index.html' '//h1' 'Module foo'
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'empty_mod_public'
pub mod foo {}

//@ has 'empty_mod_public/index.html' '//a[@href="bar/index.html"]' 'bar'
//@ hasraw 'empty_mod_public/sidebar-items.js' 'bar'
//@ matches 'empty_mod_public/bar/index.html' '//h1' 'Module bar'
//@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'empty_mod_public'
pub mod bar {
    //@ has 'empty_mod_public/bar/index.html' '//a[@href="baz/index.html"]' 'baz'
    //@ hasraw 'empty_mod_public/bar/sidebar-items.js' 'baz'
    //@ matches 'empty_mod_public/bar/baz/index.html' '//h1' 'Module baz'
    //@ matches - '//*[@class="rustdoc-breadcrumbs"]' 'empty_mod_public::bar'
    pub mod baz {}
}
