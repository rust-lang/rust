use super::{write_code, DecorationInfo};
use crate::html::format::Buffer;
use expect_test::expect_file;
use rustc_data_structures::fx::FxHashMap;
use rustc_span::create_default_session_globals_then;
use rustc_span::edition::Edition;

const STYLE: &str = r#"
<style>
.kw { color: #8959A8; }
.kw-2, .prelude-ty { color: #4271AE; }
.number, .string { color: #718C00; }
.self, .bool-val, .prelude-val, .attribute, .attribute .ident { color: #C82829; }
.macro, .macro-nonterminal { color: #3E999F; }
.lifetime { color: #B76514; }
.question-mark { color: #ff9011; }
</style>
"#;

#[test]
fn test_html_highlighting() {
    create_default_session_globals_then(|| {
        let src = include_str!("fixtures/sample.rs");
        let html = {
            let mut out = Buffer::new();
            write_code(&mut out, src, Edition::Edition2018, None, None);
            format!("{}<pre><code>{}</code></pre>\n", STYLE, out.into_inner())
        };
        expect_file!["fixtures/sample.html"].assert_eq(&html);
    });
}

#[test]
fn test_dos_backline() {
    create_default_session_globals_then(|| {
        let src = "pub fn foo() {\r\n\
    println!(\"foo\");\r\n\
}\r\n";
        let mut html = Buffer::new();
        write_code(&mut html, src, Edition::Edition2018, None, None);
        expect_file!["fixtures/dos_line.html"].assert_eq(&html.into_inner());
    });
}

#[test]
fn test_keyword_highlight() {
    create_default_session_globals_then(|| {
        let src = "use crate::a::foo;
use self::whatever;
let x = super::b::foo;
let y = Self::whatever;";

        let mut html = Buffer::new();
        write_code(&mut html, src, Edition::Edition2018, None, None);
        expect_file!["fixtures/highlight.html"].assert_eq(&html.into_inner());
    });
}

#[test]
fn test_union_highlighting() {
    create_default_session_globals_then(|| {
        let src = include_str!("fixtures/union.rs");
        let mut html = Buffer::new();
        write_code(&mut html, src, Edition::Edition2018, None, None);
        expect_file!["fixtures/union.html"].assert_eq(&html.into_inner());
    });
}

#[test]
fn test_decorations() {
    create_default_session_globals_then(|| {
        let src = "let x = 1;
let y = 2;";
        let mut decorations = FxHashMap::default();
        decorations.insert("example", vec![(0, 10)]);

        let mut html = Buffer::new();
        write_code(&mut html, src, Edition::Edition2018, None, Some(DecorationInfo(decorations)));
        expect_file!["fixtures/decorations.html"].assert_eq(&html.into_inner());
    });
}
