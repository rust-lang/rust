use std::fmt::{self, Display};
use std::path::PathBuf;

use askama::Template;
use rustc_data_structures::fx::FxIndexMap;

use super::static_files::{STATIC_FILES, StaticFiles};
use crate::externalfiles::ExternalHtml;
use crate::html::render::{StylePath, ensure_trailing_slash};

#[derive(Clone)]
pub(crate) struct Layout {
    pub(crate) logo: String,
    pub(crate) favicon: String,
    pub(crate) external_html: ExternalHtml,
    pub(crate) default_settings: FxIndexMap<String, String>,
    pub(crate) krate: String,
    pub(crate) krate_version: String,
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
    pub(crate) rust_logo: bool,
}

impl Page<'_> {
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
    rust_channel: &'static str,
    pub(crate) rustdoc_version: &'a str,
    // same as layout.krate, except on top-level pages like
    // Settings, Help, All Crates, and About Scraped Examples,
    // where these things instead give Rustdoc name and version.
    //
    // These are separate from the variables used for the search
    // engine, because "Rustdoc" isn't necessarily a crate in
    // the current workspace.
    display_krate: &'a str,
    display_krate_with_trailing_slash: String,
    display_krate_version_number: &'a str,
    display_krate_version_extra: &'a str,
}

pub(crate) use crate::html::render::sidebar::filters;

/// Implements [`Display`] for a function that accepts a mutable reference to a [`String`], and (optionally) writes to it.
///
/// The wrapped function will receive an empty string, and can modify it,
/// and the `Display` implementation will write the contents of the string after the function has finished.
pub(crate) struct BufDisplay<F>(pub F);

impl<F> Display for BufDisplay<F>
where
    F: Fn(&mut String),
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = String::new();
        self.0(&mut buf);
        f.write_str(&buf)
    }
}

pub(crate) fn render<T: Display, S: Display>(
    layout: &Layout,
    page: &Page<'_>,
    sidebar: S,
    t: T,
    style_files: &[StylePath],
) -> String {
    let rustdoc_version = rustc_interface::util::version_str!().unwrap_or("unknown version");

    let (display_krate, display_krate_version, display_krate_with_trailing_slash) =
        if page.root_path == "./" {
            // top level pages use Rust branding
            ("Rustdoc", rustdoc_version, String::new())
        } else {
            let display_krate_with_trailing_slash =
                ensure_trailing_slash(&layout.krate).to_string();
            (&layout.krate[..], &layout.krate_version[..], display_krate_with_trailing_slash)
        };
    let static_root_path = page.get_static_root_path();

    // bootstrap passes in parts of the version separated by tabs, but other stuff might use spaces
    let (display_krate_version_number, display_krate_version_extra) =
        display_krate_version.split_once([' ', '\t']).unwrap_or((display_krate_version, ""));

    let mut themes: Vec<String> = style_files.iter().map(|s| s.basename().unwrap()).collect();
    themes.sort();

    let content = t.to_string(); // Note: This must happen before making the sidebar.
    let sidebar = sidebar.to_string();
    PageLayout {
        static_root_path,
        page,
        layout,
        files: &STATIC_FILES,
        themes,
        sidebar,
        content,
        display_krate,
        display_krate_with_trailing_slash,
        display_krate_version_number,
        display_krate_version_extra,
        rust_channel: *crate::clean::utils::RUSTDOC_VERSION,
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
