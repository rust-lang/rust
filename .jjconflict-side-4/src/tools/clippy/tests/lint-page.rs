// This test checks that the lint list page stays accessible: the controls must
// keep the `<label for="...">`/`id="..."` pairings that screen readers use to
// name them. Without those the theme selector and the version filters are
// announced as unnamed widgets, which was four of the WAVE errors in #15604.
//
// The check is only a plain text search over the template, so it does not
// verify that an `id` landed on the right element. That is a known limitation,
// not a reason to drop the test: the pairings are easy to lose when the markup
// is reshuffled and nothing else in CI catches it. Do not remove or loosen
// these assertions to get an unrelated change to
// `util/gh-pages/index_template.html` through. Once the GUI test suite (#15634)
// lands this can be replaced by a check against the rendered page.

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
