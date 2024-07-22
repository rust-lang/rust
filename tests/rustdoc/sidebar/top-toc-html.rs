// ignore-tidy-linelength

#![crate_name = "foo"]
#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

//! # Basic [link](https://example.com) and *emphasis*
//!
//! This test case covers TOC entries with rich text inside.
//! Rustdoc normally supports headers with links, but for the
//! TOC, that would break the layout.
//!
//! For consistency, emphasis is also filtered out.

// @has foo/index.html
// User header
// @has - '//section[@id="TOC"]/h3' 'Sections'
// @has - '//section[@id="TOC"]/ul[@class="block top-toc"]/li/a[@href="#basic-link-and-emphasis"]' 'Basic link and emphasis'
// @count - '//section[@id="TOC"]/ul[@class="block top-toc"]/li/a[@href="#basic-link-and-emphasis"]/em' 0
// @count - '//section[@id="TOC"]/ul[@class="block top-toc"]/li/a[@href="#basic-link-and-emphasis"]/a' 0
