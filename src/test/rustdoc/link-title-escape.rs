#![allow(rustdoc::broken_intra_doc_links)]

#![crate_name = "foo"]

//! hello [foo]
//!
//! [foo]: url 'title & <stuff> & "things"'

// @hastext 'foo/index.html' 'title &amp; &lt;stuff&gt; &amp; &quot;things&quot;'
