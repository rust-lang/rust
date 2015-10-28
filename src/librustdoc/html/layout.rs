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
use html::styles;

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
    write!(dst,
r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="generator" content="rustdoc">
    <meta name="description" content="{description}">
    <meta name="keywords" content="{keywords}">

    <title>{title}</title>

    {styles}

    {favicon}
    {in_header}
</head>
<body class="rustdoc">
    <div id="style-changer">
        {divs}
    </div>
    <!--[if lte IE 8]>
    <div class="warning">
        This old browser is unsupported and will most likely display funky
        things.
    </div>
    <![endif]-->

    {before_content}

    <nav class="sidebar">
        {logo}
        {sidebar}
    </nav>

    <nav class="sub">
        <form class="search-form js-only">
            <div class="search-container">
                <input class="search-input" name="search"
                       autocomplete="off"
                       placeholder="Click or press ‘S’ to search, ‘?’ for more options…"
                       type="search">
            </div>
        </form>
    </nav>

    <section id='main' class="content {ty}">{content}</section>
    <section id='search' class="content hidden"></section>

    <section class="footer"></section>

    <aside id="help" class="hidden">
        <div>
            <h1 class="hidden">Help</h1>

            <div class="shortcuts">
                <h2>Keyboard Shortcuts</h2>

                <dl>
                    <dt>?</dt>
                    <dd>Show this help dialog</dd>
                    <dt>S</dt>
                    <dd>Focus the search field</dd>
                    <dt>&larrb;</dt>
                    <dd>Move up in search results</dd>
                    <dt>&rarrb;</dt>
                    <dd>Move down in search results</dd>
                    <dt>&#9166;</dt>
                    <dd>Go to active search result</dd>
                </dl>
            </div>

            <div class="infos">
                <h2>Search Tricks</h2>

                <p>
                    Prefix searches with a type followed by a colon (e.g.
                    <code>fn:</code>) to restrict the search to a given type.
                </p>

                <p>
                    Accepted types are: <code>fn</code>, <code>mod</code>,
                    <code>struct</code>, <code>enum</code>,
                    <code>trait</code>, <code>type</code>, <code>macro</code>,
                    and <code>const</code>.
                </p>

                <p>
                    Search functions by type signature (e.g.
                    <code>vec -> usize</code>)
                </p>
            </div>
        </div>
    </aside>

    {after_content}

    <script>
        window.rootPath = "{root_path}";
        window.currentCrate = "{krate}";
        window.playgroundUrl = "{play_url}";
    </script>
    <script src="{root_path}jquery.js"></script>
    <script src="{root_path}main.js"></script>
    {play_js}
    <script async src="{root_path}search-index.js"></script>
</body>
</html>"##,
    content   = *t,
    root_path = page.root_path,
    ty        = page.ty,
    styles    = styles::css_file_links(&page.root_path),
    divs      = styles::get_divs(),
    logo      = if layout.logo.is_empty() {
        "".to_string()
    } else {
        format!("<a href='{}{}/index.html'>\
                 <img src='{}' alt='' width='100'></a>",
                page.root_path, layout.krate,
                layout.logo)
    },
    title     = page.title,
    description = page.description,
    keywords = page.keywords,
    favicon   = if layout.favicon.is_empty() {
        "".to_string()
    } else {
        format!(r#"<link rel="shortcut icon" href="{}">"#, layout.favicon)
    },
    in_header = layout.external_html.in_header,
    before_content = layout.external_html.before_content,
    after_content = layout.external_html.after_content,
    sidebar   = *sidebar,
    krate     = layout.krate,
    play_url  = layout.playground_url,
    play_js   = if layout.playground_url.is_empty() {
        "".to_string()
    } else {
        format!(r#"<script src="{}playpen.js"></script>"#, page.root_path)
    },
    )
}

pub fn redirect(dst: &mut io::Write, url: &str) -> io::Result<()> {
    // <script> triggers a redirect before refresh, so this is fine.
    write!(dst,
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
    )
}
