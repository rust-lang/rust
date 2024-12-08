// ignore-tidy-linelength

#![crate_name = "foo"]
#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

//! # Basic [link](https://example.com), *emphasis*, **_very emphasis_** and `code`
//!
//! This test case covers TOC entries with rich text inside.
//! Rustdoc normally supports headers with links, but for the
//! TOC, that would break the layout.
//!
//! For consistency, emphasis is also filtered out.

//@ has foo/index.html
// User header
//@ has - '//section[@id="rustdoc-toc"]/h3' 'Sections'
//@ has - '//section[@id="rustdoc-toc"]/ul[@class="block top-toc"]/li/a[@href="#basic-link-emphasis-very-emphasis-and-code"]/@title' 'Basic link, emphasis, very emphasis and `code`'
//@ has - '//section[@id="rustdoc-toc"]/ul[@class="block top-toc"]/li/a[@href="#basic-link-emphasis-very-emphasis-and-code"]' 'Basic link, emphasis, very emphasis and code'
//@ count - '//section[@id="rustdoc-toc"]/ul[@class="block top-toc"]/li/a[@href="#basic-link-emphasis-very-emphasis-and-code"]/em' 0
//@ count - '//section[@id="rustdoc-toc"]/ul[@class="block top-toc"]/li/a[@href="#basic-link-emphasis-very-emphasis-and-code"]/a' 0
//@ count - '//section[@id="rustdoc-toc"]/ul[@class="block top-toc"]/li/a[@href="#basic-link-emphasis-very-emphasis-and-code"]/code' 1
//@ has - '//section[@id="rustdoc-toc"]/ul[@class="block top-toc"]/li/a[@href="#basic-link-emphasis-very-emphasis-and-code"]/code' 'code'
