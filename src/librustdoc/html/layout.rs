use std::path::PathBuf;

use rustc_data_structures::fx::FxHashMap;

use crate::externalfiles::ExternalHtml;
use crate::html::format::{Buffer, Print};
use crate::html::render::{ensure_trailing_slash, StylePath};

use askama::Template;

use super::static_files::{StaticFiles, STATIC_FILES};

#[derive(Clone)]
pub(crate) struct Layout {
    pub(crate) logo: String,
    pub(crate) favicon: String,
    pub(crate) external_html: ExternalHtml,
    pub(crate) default_settings: FxHashMap<String, String>,
    pub(crate) krate: String,
    /// The given user css file which allow to customize the generated
    /// documentation theme.
    pub(crate) css_file_extension: Option<PathBuf>,
    /// If true, then scrape-examples.js will be included in the output HTML file
    pub(crate) scrape_examples_extension: bool,
}

pub(crate) struct Page<'a> {
    pub(crate) title: &'a str,
    pub(crate) css_class: &'a str,
    pub(crate) root_path: &'a str,
    pub(crate) static_root_path: Option<&'a str>,
    pub(crate) description: &'a str,
    pub(crate) resource_suffix: &'a str,
}

impl<'a> Page<'a> {
    pub(crate) fn get_static_root_path(&self) -> String {
        match self.static_root_path {
            Some(s) => s.to_string(),
            None => format!("{}static.files/", self.root_path),
        }
    }
}

#[derive(Template)]
#[template(path = "page.html")]
struct PageLayout<'a> {
    static_root_path: String,
    page: &'a Page<'a>,
    layout: &'a Layout,

    files: &'static StaticFiles,

    themes: Vec<String>,
    sidebar: String,
    content: String,
    krate_with_trailing_slash: String,
    rust_channel: &'static str,
    pub(crate) rustdoc_version: &'a str,
}

pub(crate) fn render<T: Print, S: Print>(
    layout: &Layout,
    page: &Page<'_>,
    sidebar: S,
    t: T,
    style_files: &[StylePath],
) -> String {
    let static_root_path = page.get_static_root_path();
    let krate_with_trailing_slash = ensure_trailing_slash(&layout.krate).to_string();
    let mut themes: Vec<String> = style_files.iter().map(|s| s.basename().unwrap()).collect();
    themes.sort();

    let rustdoc_version = rustc_interface::util::version_str!().unwrap_or("unknown version");
    let content = Buffer::html().to_display(t); // Note: This must happen before making the sidebar.
    let sidebar = Buffer::html().to_display(sidebar);
    PageLayout {
        static_root_path,
        page,
        layout,
        files: &STATIC_FILES,
        themes,
        sidebar,
        content,
        krate_with_trailing_slash,
        rust_channel: *crate::clean::utils::DOC_CHANNEL,
        rustdoc_version,
    }
    .render()
    .unwrap()
}

pub(crate) fn redirect(url: &str) -> String {
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
