// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::io::prelude::*;
use std::io;

use externalfiles::ExternalHtml;

#[derive(Clone)]
pub struct Layout {
    pub logo: String,
    pub favicon: String,
    pub external_html: ExternalHtml,
    pub krate: String,
    pub playground_url: String,
}

pub struct Page<'a> {
    pub title: &'a str,
    pub ty: &'a str,
    pub root_path: &'a str,
    pub description: &'a str,
    pub keywords: &'a str
}

pub fn render<T: fmt::Display, S: fmt::Display>(
    dst: &mut io::Write, layout: &Layout, page: &Page, sidebar: &S, t: &T)
    -> io::Result<()>
{
    write!(dst, "{}", *t)
}

pub fn redirect(dst: &mut io::Write, url: &str) -> io::Result<()> {
    // <script> triggers a redirect before refresh, so this is fine.
    /*write!(dst,
r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta http-equiv="refresh" content="0;URL={url}">
</head>
<body>
    <p>Redirecting to <a href="{url}">{url}</a>...</p>
    <script>location.replace("{url}" + location.search + location.hash);</script>
</body>
</html>"##,
    url = url,
    )*/
    Ok(())
}
