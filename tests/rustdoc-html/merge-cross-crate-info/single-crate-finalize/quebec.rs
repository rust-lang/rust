//@ doc-flags:--merge=finalize
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ has index.html
//@ has index.html '//h1' 'List of all crates'
//@ has index.html '//ul[@class="all-items"]//a[@href="quebec/index.html"]' 'quebec'
//@ has quebec/struct.Quebec.html
//@ hasraw search.index/name/*.js 'Quebec'

// there is nothing to read from the output directory if we use a single
// crate
pub struct Quebec;
