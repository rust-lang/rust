use std::fmt;
use std::io;
use std::path::PathBuf;

use crate::externalfiles::ExternalHtml;
use crate::html::render::SlashChecker;

#[derive(Clone)]
pub struct Layout {
    pub logo: String,
    pub favicon: String,
    pub external_html: ExternalHtml,
    pub krate: String,
}

pub struct Page<'a> {
    pub title: &'a str,
    pub css_class: &'a str,
    pub root_path: &'a str,
    pub static_root_path: Option<&'a str>,
    pub description: &'a str,
    pub keywords: &'a str,
    pub resource_suffix: &'a str,
    pub extra_scripts: &'a [&'a str],
    pub static_extra_scripts: &'a [&'a str],
}

pub fn render<T: fmt::Display, S: fmt::Display>(
    dst: &mut dyn io::Write,
    layout: &Layout,
    page: &Page<'_>,
    sidebar: &S,
    t: &T,
    css_file_extension: bool,
    themes: &[PathBuf],
    generate_search_filter: bool,
) -> io::Result<()> {
    let static_root_path = page.static_root_path.unwrap_or(page.root_path);
    write!(dst,
"<!DOCTYPE html>\
<html lang=\"en\">\
<head>\
    <meta charset=\"utf-8\">\
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\
    <meta name=\"generator\" content=\"rustdoc\">\
    <meta name=\"description\" content=\"{description}\">\
    <meta name=\"keywords\" content=\"{keywords}\">\
    <title>{title}</title>\
    <link rel=\"stylesheet\" type=\"text/css\" href=\"{static_root_path}normalize{suffix}.css\">\
    <link rel=\"stylesheet\" type=\"text/css\" href=\"{static_root_path}rustdoc{suffix}.css\" \
          id=\"mainThemeStyle\">\
    {themes}\
    <link rel=\"stylesheet\" type=\"text/css\" href=\"{static_root_path}dark{suffix}.css\">\
    <link rel=\"stylesheet\" type=\"text/css\" href=\"{static_root_path}light{suffix}.css\" \
          id=\"themeStyle\">\
    <script src=\"{static_root_path}storage{suffix}.js\"></script>\
    <noscript><link rel=\"stylesheet\" href=\"{static_root_path}noscript{suffix}.css\"></noscript>\
    {css_extension}\
    {favicon}\
    {in_header}\
    <style type=\"text/css\">\
    #crate-search{{background-image:url(\"{static_root_path}down-arrow{suffix}.svg\");}}\
    </style>\
</head>\
<body class=\"rustdoc {css_class}\">\
    <!--[if lte IE 8]>\
    <div class=\"warning\">\
        This old browser is unsupported and will most likely display funky \
        things.\
    </div>\
    <![endif]-->\
    {before_content}\
    <nav class=\"sidebar\">\
        <div class=\"sidebar-menu\">&#9776;</div>\
        {logo}\
        {sidebar}\
    </nav>\
    <div class=\"theme-picker\">\
        <button id=\"theme-picker\" aria-label=\"Pick another theme!\">\
            <img src=\"{static_root_path}brush{suffix}.svg\" \
                 width=\"18\" \
                 alt=\"Pick another theme!\">\
        </button>\
        <div id=\"theme-choices\"></div>\
    </div>\
    <script src=\"{static_root_path}theme{suffix}.js\"></script>\
    <nav class=\"sub\">\
        <form class=\"search-form js-only\">\
            <div class=\"search-container\">\
                <div>{filter_crates}\
                    <input class=\"search-input\" name=\"search\" \
                           autocomplete=\"off\" \
                           spellcheck=\"false\" \
                           placeholder=\"Click or press ‘S’ to search, ‘?’ for more options…\" \
                           type=\"search\">\
                </div>\
                <a id=\"settings-menu\" href=\"{root_path}settings.html\">\
                    <img src=\"{static_root_path}wheel{suffix}.svg\" \
                         width=\"18\" \
                         alt=\"Change settings\">\
                </a>\
            </div>\
        </form>\
    </nav>\
    <section id=\"main\" class=\"content\">{content}</section>\
    <section id=\"search\" class=\"content hidden\"></section>\
    <section class=\"footer\"></section>\
    <aside id=\"help\" class=\"hidden\">\
        <div>\
            <h1 class=\"hidden\">Help</h1>\
            <div class=\"shortcuts\">\
                <h2>Keyboard Shortcuts</h2>\
                <dl>\
                    <dt><kbd>?</kbd></dt>\
                    <dd>Show this help dialog</dd>\
                    <dt><kbd>S</kbd></dt>\
                    <dd>Focus the search field</dd>\
                    <dt><kbd>↑</kbd></dt>\
                    <dd>Move up in search results</dd>\
                    <dt><kbd>↓</kbd></dt>\
                    <dd>Move down in search results</dd>\
                    <dt><kbd>↹</kbd></dt>\
                    <dd>Switch tab</dd>\
                    <dt><kbd>&#9166;</kbd></dt>\
                    <dd>Go to active search result</dd>\
                    <dt><kbd>+</kbd></dt>\
                    <dd>Expand all sections</dd>\
                    <dt><kbd>-</kbd></dt>\
                    <dd>Collapse all sections</dd>\
                </dl>\
            </div>\
            <div class=\"infos\">\
                <h2>Search Tricks</h2>\
                <p>\
                    Prefix searches with a type followed by a colon (e.g., \
                    <code>fn:</code>) to restrict the search to a given type.\
                </p>\
                <p>\
                    Accepted types are: <code>fn</code>, <code>mod</code>, \
                    <code>struct</code>, <code>enum</code>, \
                    <code>trait</code>, <code>type</code>, <code>macro</code>, \
                    and <code>const</code>.\
                </p>\
                <p>\
                    Search functions by type signature (e.g., \
                    <code>vec -> usize</code> or <code>* -> vec</code>)\
                </p>\
                <p>\
                    Search multiple things at once by splitting your query with comma (e.g., \
                    <code>str,u8</code> or <code>String,struct:Vec,test</code>)\
                </p>\
            </div>\
        </div>\
    </aside>\
    {after_content}\
    <script>\
        window.rootPath = \"{root_path}\";\
        window.currentCrate = \"{krate}\";\
    </script>\
    <script src=\"{root_path}aliases{suffix}.js\"></script>\
    <script src=\"{static_root_path}main{suffix}.js\"></script>\
    {static_extra_scripts}\
    {extra_scripts}\
    <script defer src=\"{root_path}search-index{suffix}.js\"></script>\
</body>\
</html>",
    css_extension = if css_file_extension {
        format!("<link rel=\"stylesheet\" \
                       type=\"text/css\" \
                       href=\"{static_root_path}theme{suffix}.css\">",
                static_root_path = static_root_path,
                suffix=page.resource_suffix)
    } else {
        String::new()
    },
    content   = *t,
    static_root_path = static_root_path,
    root_path = page.root_path,
    css_class = page.css_class,
    logo      = {
        let p = format!("{}{}", page.root_path, layout.krate);
        let p = SlashChecker(&p);
        if layout.logo.is_empty() {
            format!("<a href='{path}index.html'>\
                     <div class='logo-container'>\
                     <img src='{static_root_path}rust-logo{suffix}.png' alt='logo'></div></a>",
                    path=p,
                    static_root_path=static_root_path,
                    suffix=page.resource_suffix)
        } else {
            format!("<a href='{}index.html'>\
                     <div class='logo-container'><img src='{}' alt='logo'></div></a>",
                    p,
                    layout.logo)
        }
    },
    title     = page.title,
    description = page.description,
    keywords = page.keywords,
    favicon   = if layout.favicon.is_empty() {
        format!(r#"<link rel="shortcut icon" href="{static_root_path}favicon{suffix}.ico">"#,
                static_root_path=static_root_path,
                suffix=page.resource_suffix)
    } else {
        format!(r#"<link rel="shortcut icon" href="{}">"#, layout.favicon)
    },
    in_header = layout.external_html.in_header,
    before_content = layout.external_html.before_content,
    after_content = layout.external_html.after_content,
    sidebar   = *sidebar,
    krate     = layout.krate,
    themes = themes.iter()
                   .filter_map(|t| t.file_stem())
                   .filter_map(|t| t.to_str())
                   .map(|t| format!(r#"<link rel="stylesheet" type="text/css" href="{}{}{}.css">"#,
                                    static_root_path,
                                    t,
                                    page.resource_suffix))
                   .collect::<String>(),
    suffix=page.resource_suffix,
    static_extra_scripts=page.static_extra_scripts.iter().map(|e| {
        format!("<script src=\"{static_root_path}{extra_script}.js\"></script>",
                static_root_path=static_root_path,
                extra_script=e)
    }).collect::<String>(),
    extra_scripts=page.extra_scripts.iter().map(|e| {
        format!("<script src=\"{root_path}{extra_script}.js\"></script>",
                root_path=page.root_path,
                extra_script=e)
    }).collect::<String>(),
    filter_crates=if generate_search_filter {
        "<select id=\"crate-search\">\
            <option value=\"All crates\">All crates</option>\
        </select>"
    } else {
        ""
    },
    )
}

pub fn redirect(dst: &mut dyn io::Write, url: &str) -> io::Result<()> {
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
