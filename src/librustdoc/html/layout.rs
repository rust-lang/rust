use std::path::PathBuf;

use rustc_data_structures::fx::FxHashMap;

use crate::error::Error;
use crate::externalfiles::ExternalHtml;
use crate::html::format::{Buffer, Print};
use crate::html::render::{ensure_trailing_slash, StylePath};

use askama::Template;

#[derive(Clone)]
crate struct Layout {
    crate logo: String,
    crate favicon: String,
    crate external_html: ExternalHtml,
    crate default_settings: FxHashMap<String, String>,
    crate krate: String,
    /// The given user css file which allow to customize the generated
    /// documentation theme.
    crate css_file_extension: Option<PathBuf>,
    /// If true, then scrape-examples.js will be included in the output HTML file
    crate scrape_examples_extension: bool,
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

impl<'a> Page<'a> {
    crate fn get_static_root_path(&self) -> &str {
        self.static_root_path.unwrap_or(self.root_path)
    }
}

#[derive(Template)]
#[template(path = "page.html")]
struct PageLayout<'a> {
    static_root_path: &'a str,
    page: &'a Page<'a>,
    layout: &'a Layout,
    themes: Vec<String>,
    sidebar: String,
    content: String,
    krate_with_trailing_slash: String,
    crate rustdoc_version: &'a str,
}

crate fn render<T: Print, S: Print>(
    layout: &Layout,
    page: &Page<'_>,
    sidebar: S,
    t: T,
    style_files: &[StylePath],
) -> String {
    let static_root_path = page.get_static_root_path();
    let krate_with_trailing_slash = ensure_trailing_slash(&layout.krate).to_string();
    let mut themes: Vec<String> = style_files
        .iter()
        .map(StylePath::basename)
        .collect::<Result<_, Error>>()
        .unwrap_or_default();
    themes.sort();
    let rustdoc_version = rustc_interface::util::version_str().unwrap_or("unknown version");
    let content = Buffer::html().to_display(t); // Note: This must happen before making the sidebar.
    let sidebar = Buffer::html().to_display(sidebar);
    PageLayout {
        static_root_path,
        page,
        layout,
        themes,
        sidebar,
        content,
        krate_with_trailing_slash,
        rustdoc_version,
    }
    .render()
    .unwrap()
}

crate fn redirect(url: &str) -> String {
    // <script> triggers a redirect before refresh, so this is fine.
    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta http-equiv="refresh" content="0;URL={url}">
    <title>Redirection</title>
</head>
<body>
    <p>Redirecting to <a href="{url}">{url}</a>...</p>
    <script>location.replace("{url}" + location.search + location.hash);</script>
</body>
</html>"##,
        url = url,
    )
}
