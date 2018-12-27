#![crate_name = "foo"]

//! hello [foo]
//!
//! [foo]: url 'title & <stuff> & "things"'

// @has 'foo/index.html' 'title &amp; &lt;stuff&gt; &amp; &quot;things&quot;'
