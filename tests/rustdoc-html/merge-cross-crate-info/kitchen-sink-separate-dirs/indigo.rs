//@ aux-build:tango.rs
//@ aux-build:romeo.rs
//@ aux-build:quebec.rs
//@ aux-build:sierra.rs
//@ build-aux-docs
//@ doc-flags:--merge=finalize
//@ doc-flags:--include-parts-dir=info/doc.parts/tango
//@ doc-flags:--include-parts-dir=info/doc.parts/romeo
//@ doc-flags:--include-parts-dir=info/doc.parts/quebec
//@ doc-flags:--include-parts-dir=info/doc.parts/sierra
//@ doc-flags:--enable-index-page
//@ doc-flags:-Zunstable-options

//@ has index.html '//h1' 'List of all crates'
//@ has index.html
//@ has index.html '//ul[@class="all-items"]//a[@href="indigo/index.html"]' 'indigo'
//@ has index.html '//ul[@class="all-items"]//a[@href="quebec/index.html"]' 'quebec'
//@ has index.html '//ul[@class="all-items"]//a[@href="romeo/index.html"]' 'romeo'
//@ has index.html '//ul[@class="all-items"]//a[@href="sierra/index.html"]' 'sierra'
//@ has index.html '//ul[@class="all-items"]//a[@href="tango/index.html"]' 'tango'
//@ !has quebec/struct.Quebec.html
//@ !has romeo/type.Romeo.html
//@ !has sierra/struct.Sierra.html
//@ !has tango/trait.Tango.html
//@ hasraw trait.impl/tango/trait.Tango.js 'struct.Sierra.html'
//@ hasraw search.index/name/*.js 'Quebec'
//@ hasraw search.index/name/*.js 'Romeo'
//@ hasraw search.index/name/*.js 'Sierra'
//@ hasraw search.index/name/*.js 'Tango'
//@ has type.impl/sierra/struct.Sierra.js
//@ hasraw type.impl/sierra/struct.Sierra.js 'Tango'
//@ hasraw type.impl/sierra/struct.Sierra.js 'Romeo'

// document everything in the default mode, there are separate out
// directories that are linked together
