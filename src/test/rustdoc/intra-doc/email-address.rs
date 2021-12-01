#![allow(rustdoc::broken_intra_doc_links)]

//! Email me at <hello@example.com>.
//! Email me at <hello-world@example.com>.
//! Email me at <hello@localhost> (this warns but will still become a link).
// @has email_address/index.html '//a[@href="mailto:hello@example.com"]' 'hello@example.com'
// @has email_address/index.html '//a[@href="mailto:hello-world@example.com"]' 'hello-world@example.com'
// @has email_address/index.html '//a[@href="mailto:hello@localhost"]' 'hello@localhost'
