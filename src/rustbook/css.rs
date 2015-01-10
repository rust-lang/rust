// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The rust-book CSS in string form.

pub static STYLE: &'static str = r#"
@import url("//static.rust-lang.org/doc/master/rust.css");

body {
    max-width:none;
}

#toc {
    position: absolute;
    left: 0px;
    top: 0px;
    bottom: 0px;
    width: 250px;
    overflow-y: auto;
    border-right: 1px solid rgba(0, 0, 0, 0.07);
    padding: 10px 10px;
    font-size: 16px;
    background: none repeat scroll 0% 0% #FFF;
    box-sizing: border-box;
}

#page-wrapper {
    position: absolute;
    overflow-y: auto;
    left: 260px;
    right: 0px;
    top: 0px;
    bottom: 0px;
    box-sizing: border-box;
    background: none repeat scroll 0% 0% #FFF;
}

#page {
    margin-left: auto;
    margin-right:auto;
    width: 750px;
}

.chapter {
    list-style: none outside none;
    padding-left: 0px;
    line-height: 30px;
}

.section {
    list-style: none outside none;
    padding-left: 20px;
    line-height: 30px;
}

.section li {
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
}

.chapter li a {
    color: #000000;
}
"#;
