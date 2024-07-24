//@ build-aux-docs
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ hasraw search-index.js 'Quebec'
//@ has index.html '//ul[@class="all-items"]//a[@href="q/index.html"]' 'q'
//@ has index.html '//h1' 'List of all crates'
//@ has index.html
//@ has q/struct.Quebec.html

// there's nothing cross-crate going on here

pub struct Quebec;
