// compile-flags: --document-private-items

// @has 'empty_mod_private/index.html' '//a[@href="foo/index.html"]' 'foo'
// @has 'empty_mod_private/sidebar-items.js' 'foo'
// @matches 'empty_mod_private/foo/index.html' '//h1' 'Module empty_mod_private::foo'
mod foo {}

// @has 'empty_mod_private/index.html' '//a[@href="bar/index.html"]' 'bar'
// @has 'empty_mod_private/sidebar-items.js' 'bar'
// @matches 'empty_mod_private/bar/index.html' '//h1' 'Module empty_mod_private::bar'
mod bar {
    // @has 'empty_mod_private/bar/index.html' '//a[@href="baz/index.html"]' 'baz'
    // @has 'empty_mod_private/bar/sidebar-items.js' 'baz'
    // @matches 'empty_mod_private/bar/baz/index.html' '//h1' 'Module empty_mod_private::bar::baz'
    mod baz {}
}
