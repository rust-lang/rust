// edition:2018

#![no_core]
#![feature(no_core)]

// @count issue_89852/index.html '//*[@class="macro"]' 2
// @has - '//*[@class="macro"]/@href' 'macro.repro.html'
#[macro_export]
macro_rules! repro {
    () => {};
}

// @!has issue_89852/macro.repro.html '//*[@class="macro"]/@content' 'repro2'
pub use crate::repro as repro2;
