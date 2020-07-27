use rustc_ast::attr::with_session_globals;
use rustc_session::parse::ParseSess;
use rustc_span::edition::Edition;
use rustc_span::FileName;

use super::Classifier;

fn highlight(src: &str) -> String {
    let mut out = vec![];

    with_session_globals(Edition::Edition2018, || {
        let sess = ParseSess::with_silent_emitter();
        let source_file = sess.source_map().new_source_file(
            FileName::Custom(String::from("rustdoc-highlighting")),
            src.to_owned(),
        );

        let mut classifier = Classifier::new(&sess, source_file);
        classifier.write_source(&mut out).unwrap();
    });

    String::from_utf8(out).unwrap()
}

#[test]
fn function() {
    assert_eq!(
        highlight("fn main() {}"),
        r#"<span class="kw">fn</span> <span class="ident">main</span>() {}"#,
    );
}

#[test]
fn statement() {
    assert_eq!(
        highlight("let foo = true;"),
        concat!(
            r#"<span class="kw">let</span> <span class="ident">foo</span> "#,
            r#"<span class="op">=</span> <span class="bool-val">true</span>;"#,
        ),
    );
}

#[test]
fn inner_attr() {
    assert_eq!(
        highlight(r##"#![crate_type = "lib"]"##),
        concat!(
            r##"<span class="attribute">#![<span class="ident">crate_type</span> "##,
            r##"<span class="op">=</span> <span class="string">&quot;lib&quot;</span>]</span>"##,
        ),
    );
}

#[test]
fn outer_attr() {
    assert_eq!(
        highlight(r##"#[cfg(target_os = "linux")]"##),
        concat!(
            r##"<span class="attribute">#[<span class="ident">cfg</span>("##,
            r##"<span class="ident">target_os</span> <span class="op">=</span> "##,
            r##"<span class="string">&quot;linux&quot;</span>)]</span>"##,
        ),
    );
}

#[test]
fn mac() {
    assert_eq!(
        highlight("mac!(foo bar)"),
        concat!(
            r#"<span class="macro">mac</span><span class="macro">!</span>("#,
            r#"<span class="ident">foo</span> <span class="ident">bar</span>)"#,
        ),
    );
}

// Regression test for #72684
#[test]
fn andand() {
    assert_eq!(highlight("&&"), r#"<span class="op">&amp;&amp;</span>"#);
}
