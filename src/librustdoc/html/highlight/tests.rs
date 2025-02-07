use expect_test::expect_file;
use rustc_data_structures::fx::FxIndexMap;
use rustc_span::create_default_session_globals_then;

use super::{DecorationInfo, write_code};

const STYLE: &str = r#"
<style>
.kw { color: #8959A8; }
.kw-2, .prelude-ty { color: #4271AE; }
.number, .string { color: #718C00; }
.self, .bool-val, .prelude-val, .attr, .attr .ident { color: #C82829; }
.macro, .macro-nonterminal { color: #3E999F; }
.lifetime { color: #B76514; }
.question-mark { color: #ff9011; }
</style>
"#;

#[test]
fn test_html_highlighting() {
    create_default_session_globals_then(|| {
        let src = include_str!("fixtures/sample.rs");
        let html = format!("{STYLE}<pre><code>{}</code></pre>\n", write_code(src, None, None));
        expect_file!["fixtures/sample.html"].assert_eq(&html);
    });
}

#[test]
fn test_dos_backline() {
    create_default_session_globals_then(|| {
        let src = "pub fn foo() {\r\n\
    println!(\"foo\");\r\n\
}\r\n";
        expect_file!["fixtures/dos_line.html"].assert_eq(&write_code(src, None, None).to_string());
    });
}

#[test]
fn test_keyword_highlight() {
    create_default_session_globals_then(|| {
        let src = "use crate::a::foo;
use self::whatever;
let x = super::b::foo;
let y = Self::whatever;";

        expect_file!["fixtures/highlight.html"].assert_eq(&write_code(src, None, None).to_string());
    });
}

#[test]
fn test_union_highlighting() {
    create_default_session_globals_then(|| {
        let src = include_str!("fixtures/union.rs");
        expect_file!["fixtures/union.html"].assert_eq(&write_code(src, None, None).to_string());
    });
}

#[test]
fn test_decorations() {
    create_default_session_globals_then(|| {
        let src = "let x = 1;
let y = 2;
let z = 3;
let a = 4;";
        let mut decorations = FxIndexMap::default();
        decorations.insert("example", vec![(0, 10), (11, 21)]);
        decorations.insert("example2", vec![(22, 32)]);

        expect_file!["fixtures/decorations.html"]
            .assert_eq(&write_code(src, None, Some(&DecorationInfo(decorations))).to_string());
    });
}
