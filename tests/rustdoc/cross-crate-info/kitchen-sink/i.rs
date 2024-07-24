//@ aux-build:q.rs
//@ aux-build:r.rs
//@ aux-build:t.rs
//@ aux-build:s.rs
//@ build-aux-docs
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ hasraw search-index.js 'Quebec'
//@ hasraw search-index.js 'Sierra'
//@ has index.html
//@ has s/struct.Sierra.html
//@ hasraw s/struct.Sierra.html 'Tango'
//@ has index.html '//ul[@class="all-items"]//a[@href="i/index.html"]' 'i'
//@ has q/struct.Quebec.html
//@ has type.impl/s/struct.Sierra.js
//@ hasraw type.impl/s/struct.Sierra.js 'Romeo'
//@ hasraw type.impl/s/struct.Sierra.js 'Tango'
//@ has index.html '//ul[@class="all-items"]//a[@href="q/index.html"]' 'q'
//@ has index.html '//ul[@class="all-items"]//a[@href="t/index.html"]' 't'
//@ has index.html '//ul[@class="all-items"]//a[@href="s/index.html"]' 's'
//@ has r/type.Romeo.html
//@ has t/trait.Tango.html
//@ hasraw search-index.js 'Romeo'
//@ hasraw trait.impl/t/trait.Tango.js 'struct.Sierra.html'
//@ has index.html '//h1' 'List of all crates'
//@ hasraw search-index.js 'Tango'
//@ has index.html '//ul[@class="all-items"]//a[@href="r/index.html"]' 'r'

// document everything in the default mode


