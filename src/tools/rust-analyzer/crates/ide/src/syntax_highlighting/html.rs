//! Renders a bit of code as HTML.

use hir::{EditionedFileId, Semantics};
use oorandom::Rand32;
use stdx::format_to;
use syntax::AstNode;

use crate::{
    FileId, RootDatabase,
    syntax_highlighting::{HighlightConfig, highlight},
};

pub(crate) fn highlight_as_html(db: &RootDatabase, file_id: FileId, rainbow: bool) -> String {
    let sema = Semantics::new(db);
    let file_id = sema
        .attach_first_edition(file_id)
        .unwrap_or_else(|| EditionedFileId::current_edition(db, file_id));
    let file = sema.parse(file_id);
    let file = file.syntax();
    fn rainbowify(seed: u64) -> String {
        let mut rng = Rand32::new(seed);
        format!(
            "hsl({h},{s}%,{l}%)",
            h = rng.rand_range(0..361),
            s = rng.rand_range(42..99),
            l = rng.rand_range(40..91),
        )
    }

    let hl_ranges = highlight(
        db,
        HighlightConfig {
            strings: true,
            punctuation: true,
            specialize_punctuation: true,
            specialize_operator: true,
            operator: true,
            inject_doc_comment: true,
            macro_bang: true,
            syntactic_name_ref_highlighting: false,
        },
        file_id.file_id(db),
        None,
    );
    let text = file.to_string();
    let mut buf = String::new();
    buf.push_str(STYLE);
    buf.push_str("<pre><code>");
    for r in &hl_ranges {
        let chunk = html_escape(&text[r.range]);
        if r.highlight.is_empty() {
            format_to!(buf, "{}", chunk);
            continue;
        }

        let class = r.highlight.to_string().replace('.', " ");
        let color = match (rainbow, r.binding_hash) {
            (true, Some(hash)) => {
                format!(" data-binding-hash=\"{hash}\" style=\"color: {};\"", rainbowify(hash))
            }
            _ => "".into(),
        };
        format_to!(buf, "<span class=\"{}\"{}>{}</span>", class, color, chunk);
    }
    buf.push_str("</code></pre>");
    buf
}

//FIXME: like, real html escaping
fn html_escape(text: &str) -> String {
    text.replace('<', "&lt;").replace('>', "&gt;")
}

const STYLE: &str = "
<style>
body                { margin: 0; }
pre                 { color: #DCDCCC; background: #3F3F3F; font-size: 22px; padding: 0.4em; }

.lifetime           { color: #DFAF8F; font-style: italic; }
.label              { color: #DFAF8F; font-style: italic; }
.comment            { color: #7F9F7F; }
.documentation      { color: #629755; }
.intra_doc_link     { font-style: italic; }
.injected           { opacity: 0.65 ; }
.struct, .enum      { color: #7CB8BB; }
.enum_variant       { color: #BDE0F3; }
.string_literal     { color: #CC9393; }
.field              { color: #94BFF3; }
.function           { color: #93E0E3; }
.parameter          { color: #94BFF3; }
.text               { color: #DCDCCC; }
.type               { color: #7CB8BB; }
.builtin_type       { color: #8CD0D3; }
.type_param         { color: #DFAF8F; }
.attribute          { color: #94BFF3; }
.numeric_literal    { color: #BFEBBF; }
.bool_literal       { color: #BFE6EB; }
.macro              { color: #94BFF3; }
.proc_macro         { color: #94BFF3; text-decoration: underline; }
.derive             { color: #94BFF3; font-style: italic; }
.module             { color: #AFD8AF; }
.value_param        { color: #DCDCCC; }
.variable           { color: #DCDCCC; }
.format_specifier   { color: #CC696B; }
.mutable            { text-decoration: underline; }
.escape_sequence    { color: #94BFF3; }
.keyword            { color: #F0DFAF; font-weight: bold; }
.control            { font-style: italic; }
.reference          { font-style: italic; font-weight: bold; }
.const              { font-weight: bolder; }
.unsafe             { color: #BC8383; }

.invalid_escape_sequence { color: #FC5555; text-decoration: wavy underline; }
.unresolved_reference    { color: #FC5555; text-decoration: wavy underline; }
</style>
";
