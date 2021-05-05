use super::write_code;
use crate::html::format::Buffer;
use expect_test::expect_file;
use rustc_span::edition::Edition;
use rustc_span::with_default_session_globals;

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
    with_default_session_globals(|| {
        let src = include_str!("fixtures/sample.rs");
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
    with_default_session_globals(|| {
        let src = "pub fn foo() {\r\n\
    println!(\"foo\");\r\n\
}\r\n";
        let mut html = Buffer::new();
        write_code(&mut html, src, Edition::Edition2018);
        expect_file!["fixtures/dos_line.html"].assert_eq(&html.into_inner());
    });
}
