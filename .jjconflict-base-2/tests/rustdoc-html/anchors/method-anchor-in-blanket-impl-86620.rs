//@ aux-build:issue-86620-1.rs
#![crate_name = "foo"]
// https://github.com/rust-lang/rust/issues/86620

extern crate issue_86620_1;

use issue_86620_1::*;

//@ !has foo/struct.S.html '//*[@id="method.vzip"]//a[@class="fnname"]/@href' #tymethod.vzip
//@ has foo/struct.S.html '//*[@id="method.vzip"]//a[@class="anchor"]/@href' #method.vzip
pub struct S;
