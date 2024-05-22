//! Tidy check to make sure light and dark themes are synchronized between
//! JS-controlled rustdoc.css and noscript.css

use std::path::Path;

pub fn check(librustdoc_path: &Path, bad: &mut bool) {
    let rustdoc_css = "html/static/css/rustdoc.css";
    let noscript_css = "html/static/css/noscript.css";
    let rustdoc_css_contents = std::fs::read_to_string(librustdoc_path.join(rustdoc_css))
        .unwrap_or_else(|e| panic!("failed to read librustdoc/{rustdoc_css}: {e}"));
    let noscript_css_contents = std::fs::read_to_string(librustdoc_path.join(noscript_css))
        .unwrap_or_else(|e| panic!("failed to read librustdoc/{noscript_css}: {e}"));
    compare_themes_from_files(
        "light",
        rustdoc_css_contents.lines().enumerate().map(|(i, l)| (i + 1, l.trim())),
        noscript_css_contents.lines().enumerate().map(|(i, l)| (i + 1, l.trim())),
        bad,
    );
    compare_themes_from_files(
        "dark",
        rustdoc_css_contents.lines().enumerate(),
        noscript_css_contents.lines().enumerate(),
        bad,
    );
}

fn compare_themes_from_files<'a>(
    name: &str,
    mut rustdoc_css_lines: impl Iterator<Item = (usize, &'a str)>,
    mut noscript_css_lines: impl Iterator<Item = (usize, &'a str)>,
    bad: &mut bool,
) {
    let begin_theme_pat = format!("/* Begin theme: {name}");
    let mut found_theme = None;
    let mut found_theme_noscript = None;
    while let Some((rustdoc_css_line_number, rustdoc_css_line)) = rustdoc_css_lines.next() {
        if !rustdoc_css_line.starts_with(&begin_theme_pat) {
            continue;
        }
        if let Some(found_theme) = found_theme {
            tidy_error!(
                bad,
                "rustdoc.css contains two {name} themes on lines {rustdoc_css_line_number} and {found_theme}",
            );
            return;
        }
        found_theme = Some(rustdoc_css_line_number);
        while let Some((noscript_css_line_number, noscript_css_line)) = noscript_css_lines.next() {
            if !noscript_css_line.starts_with(&begin_theme_pat) {
                continue;
            }
            if let Some(found_theme_noscript) = found_theme_noscript {
                tidy_error!(
                    bad,
                    "noscript.css contains two {name} themes on lines {noscript_css_line_number} and {found_theme_noscript}",
                );
                return;
            }
            found_theme_noscript = Some(noscript_css_line_number);
            compare_themes(name, &mut rustdoc_css_lines, &mut noscript_css_lines, bad);
        }
    }
}

fn compare_themes<'a>(
    name: &str,
    rustdoc_css_lines: impl Iterator<Item = (usize, &'a str)>,
    noscript_css_lines: impl Iterator<Item = (usize, &'a str)>,
    bad: &mut bool,
) {
    let end_theme_pat = format!("/* End theme: {name}");
    for (
        (rustdoc_css_line_number, rustdoc_css_line),
        (noscript_css_line_number, noscript_css_line),
    ) in rustdoc_css_lines.zip(noscript_css_lines)
    {
        if noscript_css_line.starts_with(":root, :root:not([data-theme]) {")
            && (rustdoc_css_line.starts_with(&format!(r#":root[data-theme="{name}"] {{"#))
                || rustdoc_css_line.starts_with(&format!(
                    r#":root[data-theme="{name}"], :root:not([data-theme]) {{"#
                )))
        {
            // selectors are different between rustdoc.css and noscript.css
            // that's why they both exist: one uses JS, the other uses media queries
            continue;
        }
        if noscript_css_line.starts_with(&end_theme_pat)
            && rustdoc_css_line.starts_with(&end_theme_pat)
        {
            break;
        }
        if rustdoc_css_line != noscript_css_line {
            tidy_error!(
                bad,
                "noscript.css:{noscript_css_line_number} and rustdoc.css:{rustdoc_css_line_number} contain copies of {name} theme that are not the same",
            );
            eprintln!("- {noscript_css_line}");
            eprintln!("+ {rustdoc_css_line}");
            return;
        }
    }
}
