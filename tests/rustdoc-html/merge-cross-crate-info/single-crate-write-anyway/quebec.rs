//@ doc-flags:--read-doc-meta-dir=.
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ has index.html
//@ has index.html '//h1' 'List of all crates'
//@ has index.html '//ul[@class="all-items"]//a[@href="quebec/index.html"]' 'quebec'
//@ has quebec/struct.Quebec.html
//@ hasraw search.index/name/*.js 'Quebec'

// we can --write-doc-meta-dir, but that doesn't do anything other than create
// the file
pub struct Quebec;
