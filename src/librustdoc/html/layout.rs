use std::path::PathBuf;

use rustc_data_structures::fx::FxHashMap;

use crate::externalfiles::ExternalHtml;
use crate::html::escape::Escape;
use crate::html::format::{Buffer, Print};
use crate::html::render::{ensure_trailing_slash, StylePath};

use serde::Serialize;

#[derive(Clone, Serialize)]
crate struct Layout {
    crate logo: String,
    crate favicon: String,
    crate external_html: ExternalHtml,
    crate default_settings: FxHashMap<String, String>,
    crate krate: String,
    /// The given user css file which allow to customize the generated
    /// documentation theme.
    crate css_file_extension: Option<PathBuf>,
    /// If false, the `select` element to have search filtering by crates on rendered docs
    /// won't be generated.
    crate generate_search_filter: bool,
    /// If true, then scrape-examples.js will be included in the output HTML file
    crate scrape_examples_extension: bool,
}

#[derive(Serialize)]
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

#[derive(Serialize)]
struct PageLayout<'a> {
    static_root_path: &'a str,
    page: &'a Page<'a>,
    layout: &'a Layout,
    style_files: String,
    sidebar: String,
    content: String,
    krate_with_trailing_slash: String,
}

crate fn render<T: Print, S: Print>(
    templates: &tera::Tera,
    layout: &Layout,
    page: &Page<'_>,
    sidebar: S,
    t: T,
    style_files: &[StylePath],
) -> String {
    let static_root_path = page.get_static_root_path();
    let krate_with_trailing_slash = ensure_trailing_slash(&layout.krate).to_string();
    let style_files = style_files
        .iter()
        .filter_map(|t| t.path.file_stem().map(|stem| (stem, t.disabled)))
        .filter_map(|t| t.0.to_str().map(|path| (path, t.1)))
        .map(|t| {
            format!(
                r#"<link rel="stylesheet" type="text/css" href="{}.css" {} {}>"#,
                Escape(&format!("{}{}{}", static_root_path, t.0, page.resource_suffix)),
                if t.1 { "disabled" } else { "" },
                if t.0 == "light" { "id=\"themeStyle\"" } else { "" }
            )
        })
        .collect::<String>();
    let content = Buffer::html().to_display(t); // Note: This must happen before making the sidebar.
    let sidebar = Buffer::html().to_display(sidebar);
    let teractx = tera::Context::from_serialize(PageLayout {
        static_root_path,
        page,
        layout,
        style_files,
        sidebar,
        content,
        krate_with_trailing_slash,
    })
    .unwrap();
    templates.render("page.html", &teractx).unwrap()
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
