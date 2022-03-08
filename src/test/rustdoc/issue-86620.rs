// aux-build:issue-86620-1.rs

extern crate issue_86620_1;

use issue_86620_1::*;

// @!has issue_86620/struct.S.html '//*[@id="method.vzip"]//a[@class="fnname"]/@href' #tymethod.vzip
// @has issue_86620/struct.S.html '//*[@id="method.vzip"]//a[@class="anchor"]/@href' #method.vzip
pub struct S;
