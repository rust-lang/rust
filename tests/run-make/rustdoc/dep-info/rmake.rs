// This is a simple smoke test for rustdoc's `--emit dep-info` feature. It prints out
// information about dependencies in a Makefile-compatible format, as a `.d` file.

//@ needs-target-std

use run_make_support::assertion_helpers::{assert_contains, assert_not_contains};
use run_make_support::{path, rfs, rustdoc};

fn main() {
    rfs::create_dir("doc");

    // Ensure that all kinds of input reading flags end up in dep-info.
    rustdoc()
        .input("lib.rs")
        .arg("-Zunstable-options")
        .arg("--html-before-content=before.html")
        .arg("--markdown-after-content=after.md")
        .arg("--extend-css=extend.css")
        .arg("--theme=custom_theme.css")
        .arg("--index-page=index-page.md")
        .emit("dep-info")
        .run();

    let content = rfs::read_to_string("doc/foo.d");
    assert_contains(&content, "lib.rs:");
    assert_contains(&content, "foo.rs:");
    assert_contains(&content, "bar.rs:");
    assert_contains(&content, "doc.md:");
    assert_contains(&content, "after.md:");
    assert_contains(&content, "before.html:");
    assert_contains(&content, "extend.css:");
    assert_contains(&content, "custom_theme.css:");
    assert_contains(&content, "index-page.md:");
    // Only emit dep-info. Don't emit the actual page.
    assert!(!path("doc/foo/index.html").exists());
    assert!(!path("doc/custom_theme.css").exists());
    // weird that --extend-css generates a file named theme.css
    assert!(!path("doc/theme.css").exists());

    // Now try emitting dep-info and html files at the same time.
    rustdoc()
        .input("lib.rs")
        .arg("-Zunstable-options")
        .arg("--html-before-content=before.html")
        .arg("--markdown-after-content=after.md")
        .arg("--extend-css=extend.css")
        .arg("--theme=custom_theme.css")
        .arg("--index-page=index-page.md")
        .emit("dep-info,html-non-static-files,html-static-files")
        .run();
    assert!(path("doc/foo/index.html").exists());
    // These files are copied into the doc output folder,
    // which is why they show up in dep-info.
    assert!(path("doc/custom_theme.css").exists());
    // weird that --extend-css generates a file named theme.css
    assert!(path("doc/theme.css").exists());

    // Now we check that we can provide a file name to the `dep-info` argument.
    rustdoc().input("lib.rs").arg("-Zunstable-options").emit("dep-info=bla.d").run();
    assert!(path("bla.d").exists());

    // The last emit-type wins. The same behavior as rustc.
    rustdoc()
        .input("lib.rs")
        .arg("-Zunstable-options")
        .emit("dep-info=precedence1.d")
        .emit("dep-info=precedence2.d")
        .emit("dep-info=precedence3.d")
        .run();
    assert!(!path("precedence1.d").exists());
    assert!(!path("precedence2.d").exists());
    assert!(path("precedence3.d").exists());

    // stdout (-) also wins if being the last.
    let result = rustdoc()
        .input("lib.rs")
        .arg("-Zunstable-options")
        .emit("dep-info=precedence1.d")
        .emit("dep-info=-")
        .run();
    assert!(!path("precedence1.d").exists());
    assert!(!path("-").exists()); // `-` shouldn't be treated as a file path
    assert!(!result.stdout().is_empty()); // Something emitted to stdout

    // test --emit=dep-info combined with plain markdown input
    rustdoc().input("example.md").arg("-Zunstable-options").emit("dep-info").run();
    let content = rfs::read_to_string("doc/example.d");
    assert_contains(&content, "example.md:");
    assert_not_contains(&content, "lib.rs:");
    assert_not_contains(&content, "foo.rs:");
    assert_not_contains(&content, "bar.rs:");
    assert_not_contains(&content, "doc.md:");
    assert_not_contains(&content, "after.md:");
    assert_not_contains(&content, "before.html:");
    assert_not_contains(&content, "extend.css:");
    assert_not_contains(&content, "custom_theme.css:");
    // Only emit dep-info, not the actual html.
    assert!(!path("doc/example.html").exists());

    // combine --emit=dep-info=filename with plain markdown input
    rustdoc()
        .input("example.md")
        .arg("-Zunstable-options")
        .arg("--html-before-content=before.html")
        .arg("--markdown-after-content=after.md")
        .arg("--extend-css=extend.css")
        .arg("--theme=custom_theme.css")
        .arg("--markdown-css=markdown.css")
        .arg("--index-page=index-page.md")
        .emit("dep-info=example.d,html-non-static-files,html-static-files")
        .run();
    assert!(path("doc/example.html").exists());
    let content = rfs::read_to_string("example.d");
    assert_contains(&content, "example.md:");
    assert_not_contains(&content, "lib.rs:");
    assert_not_contains(&content, "foo.rs:");
    assert_not_contains(&content, "bar.rs:");
    assert_not_contains(&content, "doc.md:");
    assert_contains(&content, "after.md:");
    assert_contains(&content, "before.html:");
    assert_contains(&content, "index-page.md:");
    // This is a hotlink, not a file that gets copied,
    // so it shouldn't add to the dep-info, it shouldn't be copied,
    // and it shouldn't be resolved relative to the root path.
    //
    // It's weird that this is different from the other two css
    // files, but it's stable, so I can't change it.
    assert!(!path("doc/markdown.css").exists());
    assert_not_contains(&content, "markdown.css:");
    // These files aren't actually used, and the fact that they show up
    // is arguably a bug, but test it anyway.
    assert_contains(&content, "extend.css:");
    assert_contains(&content, "custom_theme.css:");
}
