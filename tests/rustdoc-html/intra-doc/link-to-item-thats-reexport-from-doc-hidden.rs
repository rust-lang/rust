//@ aux-build:doc-hidden-reexported.rs
//@ build-aux-docs

//@ has doc_hidden_reexported/docs/io_uring_provided_buffers/index.html
//@ matchesraw - '<meta http-equiv="refresh" content="0;URL=../../../doc_hidden_reexported/io_uring_provided_buffers/index.html">'
//@ has doc_hidden_reexported/io_uring_provided_buffers/index.html

//@ has doc_hidden_reexported/docs/struct.IoUringProvidedBuffers.html
//@ matchesraw - '<meta http-equiv="refresh" content="0;URL=../../doc_hidden_reexported/struct.IoUringProvidedBuffers.html">'
//@ has doc_hidden_reexported/struct.IoUringProvidedBuffers.html

//! Link to [doc_hidden_reexported::io_uring_provided_buffers] and
//! [doc_hidden_reexported::IoUringProvidedBuffers]
//!
//! Because these cross-crate links point at the internal, private path,
//! we need to make sure the redirect gets generated.

//@ has link_to_item_thats_reexport_from_doc_hidden/index.html
//@ count - '//a[@href="../doc_hidden_reexported/docs/io_uring_provided_buffers/index.html"]' 1
//@ count - '//a[@href="../doc_hidden_reexported/docs/struct.IoUringProvidedBuffers.html"]' 1

extern crate doc_hidden_reexported;
