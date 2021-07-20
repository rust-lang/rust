use super::write_code;
use crate::html::format::Buffer;
use crate::html::markdown::Line;
use expect_test::expect_file;
use rustc_span::create_default_session_globals_then;
use rustc_span::edition::Edition;

use std::borrow::Cow;
use std::iter::once;

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
        let src = once(Line::Shown(Cow::Borrowed(src)));
        let html = {
            let mut out = Buffer::new();
            write_code(&mut out, src, Edition::Edition2018);
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
        let src = once(Line::Shown(Cow::Borrowed(src)));
        let mut html = Buffer::new();
        write_code(&mut html, src, Edition::Edition2018);
        expect_file!["fixtures/dos_line.html"].assert_eq(&html.into_inner());
    });
}
