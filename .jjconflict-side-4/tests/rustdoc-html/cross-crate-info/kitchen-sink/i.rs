//@ aux-build:r.rs
//@ aux-build:q.rs
//@ aux-build:t.rs
//@ aux-build:s.rs
//@ build-aux-docs
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ has index.html '//h1' 'List of all crates'
//@ has index.html
//@ has index.html '//ul[@class="all-items"]//a[@href="i/index.html"]' 'i'
//@ has index.html '//ul[@class="all-items"]//a[@href="q/index.html"]' 'q'
//@ has index.html '//ul[@class="all-items"]//a[@href="r/index.html"]' 'r'
//@ has index.html '//ul[@class="all-items"]//a[@href="s/index.html"]' 's'
//@ has index.html '//ul[@class="all-items"]//a[@href="t/index.html"]' 't'
//@ has q/struct.Quebec.html
//@ has r/type.Romeo.html
//@ has s/struct.Sierra.html
//@ has t/trait.Tango.html
//@ hasraw s/struct.Sierra.html 'Tango'
//@ hasraw trait.impl/t/trait.Tango.js 'struct.Sierra.html'
//@ hasraw search.index/name/*.js 'Quebec'
//@ hasraw search.index/name/*.js 'Romeo'
//@ hasraw search.index/name/*.js 'Sierra'
//@ hasraw search.index/name/*.js 'Tango'
//@ has type.impl/s/struct.Sierra.js
//@ hasraw type.impl/s/struct.Sierra.js 'Tango'
//@ hasraw type.impl/s/struct.Sierra.js 'Romeo'

// document everything in the default mode
