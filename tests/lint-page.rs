#[test]
fn lint_page_accessibility() {
    let template = include_str!("../util/gh-pages/index_template.html");

    // The theme selector must have a programmatically associated label
    assert!(
        template.contains(r#"for="theme-choice""#),
        "theme-choice <select> is missing an associated <label>",
    );

    // The version-filter inputs are generated via a template loop.
    // Each input must carry an id and its sibling <label> a matching `for`.
    assert!(
        template.contains(r#"id="version-filter-{{ name }}""#),
        "version-filter inputs are missing id=\"version-filter-...\"",
    );
    assert!(
        template.contains(r#"for="version-filter-{{ name }}""#),
        "version-filter inputs are missing an associated <label for=\"version-filter-...\">",
    );
}
