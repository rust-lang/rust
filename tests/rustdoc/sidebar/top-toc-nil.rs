#![crate_name = "foo"]

//! This test case covers missing top TOC entries.

// @has foo/index.html
// User header
// @!has - '//section[@id="TOC"]/ul[@class="block top-toc"]' 'Basic link and emphasis'
