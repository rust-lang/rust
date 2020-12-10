use std::collections::HashMap;
use std::path::PathBuf;

use crate::externalfiles::ExternalHtml;
use crate::html::escape::Escape;
use crate::html::format::{Buffer, Print};
use crate::html::render::{ensure_trailing_slash, StylePath};

#[derive(Clone)]
crate struct Layout {
    crate logo: String,
    crate favicon: String,
    crate external_html: ExternalHtml,
    crate default_settings: HashMap<String, String>,
    crate krate: String,
    /// The given user css file which allow to customize the generated
    /// documentation theme.
    crate css_file_extension: Option<PathBuf>,
    /// If false, the `select` element to have search filtering by crates on rendered docs
    /// won't be generated.
    crate generate_search_filter: bool,
}

crate struct Page<'a> {
    crate title: &'a str,
    crate css_class: &'a str,
    crate root_path: &'a str,
    crate static_root_path: Option<&'a str>,
    crate description: &'a str,
    crate keywords: &'a str,
    crate resource_suffix: &'a str,
    crate extra_scripts: &'a [&'a str],
    crate static_extra_scripts: &'a [&'a str],
}

crate fn render<T: Print, S: Print>(
    layout: &Layout,
    page: &Page<'_>,
    sidebar: S,
    t: T,
    style_files: &[StylePath],
) -> String {
    let static_root_path = page.static_root_path.unwrap_or(page.root_path);
    format!(
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
    {style_files}\
    <script id=\"default-settings\"{default_settings}></script>\
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
        <button id=\"theme-picker\" aria-label=\"Pick another theme!\" aria-haspopup=\"menu\">\
            <img src=\"{static_root_path}brush{suffix}.svg\" \
                 width=\"18\" \
                 alt=\"Pick another theme!\">\
        </button>\
        <div id=\"theme-choices\" role=\"menu\"></div>\
    </div>\
    <script src=\"{static_root_path}theme{suffix}.js\"></script>\
    <nav class=\"sub\">\
        <form class=\"search-form\">\
            <div class=\"search-container\">\
                <div>{filter_crates}\
                    <input class=\"search-input\" name=\"search\" \
                           disabled \
                           autocomplete=\"off\" \
                           spellcheck=\"false\" \
                           placeholder=\"Click or press ‘S’ to search, ‘?’ for more options…\" \
                           type=\"search\">\
                </div>\
                <button class=\"help-button\">?</button>
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
    {after_content}\
    <script>\
        window.rootPath = \"{root_path}\";\
        window.currentCrate = \"{krate}\";\
    </script>\
    <script src=\"{static_root_path}main{suffix}.js\"></script>\
    {static_extra_scripts}\
    {extra_scripts}\
    <script defer src=\"{root_path}search-index{suffix}.js\"></script>\
</body>\
</html>",
        css_extension = if layout.css_file_extension.is_some() {
            format!(
                "<link rel=\"stylesheet\" \
                       type=\"text/css\" \
                       href=\"{static_root_path}theme{suffix}.css\">",
                static_root_path = static_root_path,
                suffix = page.resource_suffix
            )
        } else {
            String::new()
        },
        content = Buffer::html().to_display(t),
        static_root_path = static_root_path,
        root_path = page.root_path,
        css_class = page.css_class,
        logo = {
            let p = format!("{}{}", page.root_path, layout.krate);
            let p = ensure_trailing_slash(&p);
            if layout.logo.is_empty() {
                format!(
                    "<a href='{path}index.html'>\
                     <div class='logo-container rust-logo'>\
                     <img src='{static_root_path}rust-logo{suffix}.png' alt='logo'></div></a>",
                    path = p,
                    static_root_path = static_root_path,
                    suffix = page.resource_suffix
                )
            } else {
                format!(
                    "<a href='{}index.html'>\
                     <div class='logo-container'><img src='{}' alt='logo'></div></a>",
                    p, layout.logo
                )
            }
        },
        title = page.title,
        description = page.description,
        keywords = page.keywords,
        favicon = if layout.favicon.is_empty() {
            format!(
                r##"<link rel="icon" type="image/svg+xml" href="{static_root_path}favicon{suffix}.svg">
<link rel="alternate icon" type="image/png" href="{static_root_path}favicon-16x16{suffix}.png">
<link rel="alternate icon" type="image/png" href="{static_root_path}favicon-32x32{suffix}.png">"##,
                static_root_path = static_root_path,
                suffix = page.resource_suffix
            )
        } else {
            format!(r#"<link rel="shortcut icon" href="{}">"#, layout.favicon)
        },
        in_header = layout.external_html.in_header,
        before_content = layout.external_html.before_content,
        after_content = layout.external_html.after_content,
        sidebar = Buffer::html().to_display(sidebar),
        krate = layout.krate,
        default_settings = layout
            .default_settings
            .iter()
            .map(|(k, v)| format!(r#" data-{}="{}""#, k.replace('-', "_"), Escape(v)))
            .collect::<String>(),
        style_files = style_files
            .iter()
            .filter_map(|t| {
                if let Some(stem) = t.path.file_stem() { Some((stem, t.disabled)) } else { None }
            })
            .filter_map(|t| {
                if let Some(path) = t.0.to_str() { Some((path, t.1)) } else { None }
            })
            .map(|t| format!(
                r#"<link rel="stylesheet" type="text/css" href="{}.css" {} {}>"#,
                Escape(&format!("{}{}{}", static_root_path, t.0, page.resource_suffix)),
                if t.1 { "disabled" } else { "" },
                if t.0 == "light" { "id=\"themeStyle\"" } else { "" }
            ))
            .collect::<String>(),
        suffix = page.resource_suffix,
        static_extra_scripts = page
            .static_extra_scripts
            .iter()
            .map(|e| {
                format!(
                    "<script src=\"{static_root_path}{extra_script}.js\"></script>",
                    static_root_path = static_root_path,
                    extra_script = e
                )
            })
            .collect::<String>(),
        extra_scripts = page
            .extra_scripts
            .iter()
            .map(|e| {
                format!(
                    "<script src=\"{root_path}{extra_script}.js\"></script>",
                    root_path = page.root_path,
                    extra_script = e
                )
            })
            .collect::<String>(),
        filter_crates = if layout.generate_search_filter {
            "<select id=\"crate-search\">\
                 <option value=\"All crates\">All crates</option>\
             </select>"
        } else {
            ""
        },
    )
}

crate fn redirect(url: &str) -> String {
    // <script> triggers a redirect before refresh, so this is fine.
    format!(
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
