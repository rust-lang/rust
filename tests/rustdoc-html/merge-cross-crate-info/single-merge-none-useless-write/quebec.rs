//@ doc-flags:--merge=none
//@ doc-flags:--parts-out-dir=info/doc.parts/quebec
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ !has index.html
//@ has quebec/struct.Quebec.html
//@ !has search.index/name/*.js

// --merge=none doesn't write anything, despite --parts-out-dir
pub struct Quebec;
