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
use std::io;

#[deriving(Clone)]
pub struct Layout {
    logo: ~str,
    favicon: ~str,
    krate: ~str,
}

pub struct Page<'a> {
    title: &'a str,
    ty: &'a str,
    root_path: &'a str,
}

pub fn render<T: fmt::Show, S: fmt::Show>(
    dst: &mut io::Writer, layout: &Layout, page: &Page, sidebar: &S, t: &T)
    -> fmt::Result
{
    write!(dst,
r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>{title}</title>

    <link href='http://fonts.googleapis.com/css?family=Oswald:700|Inconsolata:400,700'
          rel='stylesheet' type='text/css'>
    <link rel="stylesheet" type="text/css" href="{root_path}main.css">

    {favicon, select, none{} other{<link rel="shortcut icon" href="\#" />}}
</head>
<body>
    <!--[if lte IE 8]>
    <div class="warning">
        This old browser is unsupported and will most likely display funky
        things.
    </div>
    <![endif]-->

    <section class="sidebar">
        {logo, select, none{} other{
            <a href='{root_path}{krate}/index.html'><img src='\#' alt=''/></a>
        }}

        {sidebar}
    </section>

    <nav class="sub">
        <form class="search-form js-only">
            <button class="do-search">Search</button>
            <div class="search-container">
                <input class="search-input" name="search"
                       autocomplete="off"
                       placeholder="Search documentation..."
                       type="search" />
            </div>
        </form>
    </nav>

    <section id='main' class="content {ty}">{content}</section>
    <section id='search' class="content hidden"></section>

    <section class="footer"></section>

    <div id="help" class="hidden">
        <div class="shortcuts">
            <h1>Keyboard shortcuts</h1>
            <dl>
                <dt>?</dt>
                <dd>Show this help dialog</dd>
                <dt>S</dt>
                <dd>Focus the search field</dd>
                <dt>&uarr;</dt>
                <dd>Move up in search results</dd>
                <dt>&darr;</dt>
                <dd>Move down in search results</dd>
                <dt>&\#9166;</dt>
                <dd>Go to active search result</dd>
            </dl>
        </div>
        <div class="infos">
            <h1>Search tricks</h1>
            <p>
                Prefix searches with a type followed by a colon (e.g.
                <code>fn:</code>) to restrict the search to a given type.
            </p>
            <p>
                Accepted types are: <code>fn</code>, <code>mod</code>,
                <code>struct</code> (or <code>str</code>), <code>enum</code>,
                <code>trait</code>, <code>typedef</code> (or
                <code>tdef</code>).
            </p>
        </div>
    </div>

    <script>
        var rootPath = "{root_path}";
        var currentCrate = "{krate}";
    </script>
    <script src="{root_path}jquery.js"></script>
    <script src="{root_path}main.js"></script>
    <script async src="{root_path}search-index.js"></script>
</body>
</html>
"##,
    content   = *t,
    root_path = page.root_path,
    ty        = page.ty,
    logo      = nonestr(layout.logo),
    title     = page.title,
    favicon   = nonestr(layout.favicon),
    sidebar   = *sidebar,
    krate     = layout.krate,
    )
}

fn nonestr<'a>(s: &'a str) -> &'a str {
    if s == "" { "none" } else { s }
}
