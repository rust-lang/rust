//@ build-aux-docs
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ has index.html
//@ has index.html '//h1' 'List of all crates'
//@ has index.html '//ul[@class="all-items"]//a[@href="q/index.html"]' 'q'
//@ has q/struct.Quebec.html
//@ hasraw search.index/name/*.js 'Quebec'

// there's nothing cross-crate going on here
pub struct Quebec;
