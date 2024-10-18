use std::mem;
use std::ops::Range;

use pulldown_cmark::{
    BrokenLink, BrokenLinkCallback, CowStr, Event, LinkType, Options, Parser, Tag,
};
use rustc_ast as ast;
use rustc_ast::attr::AttributeExt;
use rustc_ast::util::comments::beautify_doc_string;
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::symbol::{Symbol, kw, sym};
use rustc_span::{DUMMY_SP, InnerSpan, Span};
use tracing::{debug, trace};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DocFragmentKind {
    /// A doc fragment created from a `///` or `//!` doc comment.
    SugaredDoc,
    /// A doc fragment created from a "raw" `#[doc=""]` attribute.
    RawDoc,
}

/// A portion of documentation, extracted from a `#[doc]` attribute.
///
/// Each variant contains the line number within the complete doc-comment where the fragment
/// starts, as well as the Span where the corresponding doc comment or attribute is located.
///
/// Included files are kept separate from inline doc comments so that proper line-number
/// information can be given when a doctest fails. Sugared doc comments and "raw" doc comments are
/// kept separate because of issue #42760.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DocFragment {
    pub span: Span,
    /// The item this doc-comment came from.
    /// Used to determine the scope in which doc links in this fragment are resolved.
    /// Typically filled for reexport docs when they are merged into the docs of the
    /// original reexported item.
    /// If the id is not filled, which happens for the original reexported item, then
    /// it has to be taken from somewhere else during doc link resolution.
    pub item_id: Option<DefId>,
    pub doc: Symbol,
    pub kind: DocFragmentKind,
    pub indent: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum MalformedGenerics {
    /// This link has unbalanced angle brackets.
    ///
    /// For example, `Vec<T` should trigger this, as should `Vec<T>>`.
    UnbalancedAngleBrackets,
    /// The generics are not attached to a type.
    ///
    /// For example, `<T>` should trigger this.
    ///
    /// This is detected by checking if the path is empty after the generics are stripped.
    MissingType,
    /// The link uses fully-qualified syntax, which is currently unsupported.
    ///
    /// For example, `<Vec as IntoIterator>::into_iter` should trigger this.
    ///
    /// This is detected by checking if ` as ` (the keyword `as` with spaces around it) is inside
    /// angle brackets.
    HasFullyQualifiedSyntax,
    /// The link has an invalid path separator.
    ///
    /// For example, `Vec:<T>:new()` should trigger this. Note that `Vec:new()` will **not**
    /// trigger this because it has no generics and thus [`strip_generics_from_path`] will not be
    /// called.
    ///
    /// Note that this will also **not** be triggered if the invalid path separator is inside angle
    /// brackets because rustdoc mostly ignores what's inside angle brackets (except for
    /// [`HasFullyQualifiedSyntax`](MalformedGenerics::HasFullyQualifiedSyntax)).
    ///
    /// This is detected by checking if there is a colon followed by a non-colon in the link.
    InvalidPathSeparator,
    /// The link has too many angle brackets.
    ///
    /// For example, `Vec<<T>>` should trigger this.
    TooManyAngleBrackets,
    /// The link has empty angle brackets.
    ///
    /// For example, `Vec<>` should trigger this.
    EmptyAngleBrackets,
}

/// Removes excess indentation on comments in order for the Markdown
/// to be parsed correctly. This is necessary because the convention for
/// writing documentation is to provide a space between the /// or //! marker
/// and the doc text, but Markdown is whitespace-sensitive. For example,
/// a block of text with four-space indentation is parsed as a code block,
/// so if we didn't unindent comments, these list items
///
/// /// A list:
/// ///
/// ///    - Foo
/// ///    - Bar
///
/// would be parsed as if they were in a code block, which is likely not what the user intended.
pub fn unindent_doc_fragments(docs: &mut [DocFragment]) {
    // `add` is used in case the most common sugared doc syntax is used ("/// "). The other
    // fragments kind's lines are never starting with a whitespace unless they are using some
    // markdown formatting requiring it. Therefore, if the doc block have a mix between the two,
    // we need to take into account the fact that the minimum indent minus one (to take this
    // whitespace into account).
    //
    // For example:
    //
    // /// hello!
    // #[doc = "another"]
    //
    // In this case, you want "hello! another" and not "hello!  another".
    let add = if docs.windows(2).any(|arr| arr[0].kind != arr[1].kind)
        && docs.iter().any(|d| d.kind == DocFragmentKind::SugaredDoc)
    {
        // In case we have a mix of sugared doc comments and "raw" ones, we want the sugared one to
        // "decide" how much the minimum indent will be.
        1
    } else {
        0
    };

    // `min_indent` is used to know how much whitespaces from the start of each lines must be
    // removed. Example:
    //
    // ///     hello!
    // #[doc = "another"]
    //
    // In here, the `min_indent` is 1 (because non-sugared fragment are always counted with minimum
    // 1 whitespace), meaning that "hello!" will be considered a codeblock because it starts with 4
    // (5 - 1) whitespaces.
    let Some(min_indent) = docs
        .iter()
        .map(|fragment| {
            fragment
                .doc
                .as_str()
                .lines()
                .filter(|line| line.chars().any(|c| !c.is_whitespace()))
                .map(|line| {
                    // Compare against either space or tab, ignoring whether they are
                    // mixed or not.
                    let whitespace = line.chars().take_while(|c| *c == ' ' || *c == '\t').count();
                    whitespace
                        + (if fragment.kind == DocFragmentKind::SugaredDoc { 0 } else { add })
                })
                .min()
                .unwrap_or(usize::MAX)
        })
        .min()
    else {
        return;
    };

    for fragment in docs {
        if fragment.doc == kw::Empty {
            continue;
        }

        let indent = if fragment.kind != DocFragmentKind::SugaredDoc && min_indent > 0 {
            min_indent - add
        } else {
            min_indent
        };

        fragment.indent = indent;
    }
}

/// The goal of this function is to apply the `DocFragment` transformation that is required when
/// transforming into the final Markdown, which is applying the computed indent to each line in
/// each doc fragment (a `DocFragment` can contain multiple lines in case of `#[doc = ""]`).
///
/// Note: remove the trailing newline where appropriate
pub fn add_doc_fragment(out: &mut String, frag: &DocFragment) {
    if frag.doc == kw::Empty {
        out.push('\n');
        return;
    }
    let s = frag.doc.as_str();
    let mut iter = s.lines();

    while let Some(line) = iter.next() {
        if line.chars().any(|c| !c.is_whitespace()) {
            assert!(line.len() >= frag.indent);
            out.push_str(&line[frag.indent..]);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
}

pub fn attrs_to_doc_fragments<'a, A: AttributeExt + Clone + 'a>(
    attrs: impl Iterator<Item = (&'a A, Option<DefId>)>,
    doc_only: bool,
) -> (Vec<DocFragment>, Vec<A>) {
    let mut doc_fragments = Vec::new();
    let mut other_attrs = Vec::<A>::new();
    for (attr, item_id) in attrs {
        if let Some((doc_str, comment_kind)) = attr.doc_str_and_comment_kind() {
            let doc = beautify_doc_string(doc_str, comment_kind);
            let (span, kind) = if attr.is_doc_comment() {
                (attr.span(), DocFragmentKind::SugaredDoc)
            } else {
                (
                    attr.value_span()
                        .map(|i| i.with_ctxt(attr.span().ctxt()))
                        .unwrap_or(attr.span()),
                    DocFragmentKind::RawDoc,
                )
            };
            let fragment = DocFragment { span, doc, kind, item_id, indent: 0 };
            doc_fragments.push(fragment);
        } else if !doc_only {
            other_attrs.push(attr.clone());
        }
    }

    unindent_doc_fragments(&mut doc_fragments);

    (doc_fragments, other_attrs)
}

/// Return the doc-comments on this item, grouped by the module they came from.
/// The module can be different if this is a re-export with added documentation.
///
/// The last newline is not trimmed so the produced strings are reusable between
/// early and late doc link resolution regardless of their position.
pub fn prepare_to_doc_link_resolution(
    doc_fragments: &[DocFragment],
) -> FxIndexMap<Option<DefId>, String> {
    let mut res = FxIndexMap::default();
    for fragment in doc_fragments {
        let out_str = res.entry(fragment.item_id).or_default();
        add_doc_fragment(out_str, fragment);
    }
    res
}

/// Options for rendering Markdown in the main body of documentation.
pub fn main_body_opts() -> Options {
    Options::ENABLE_TABLES
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_SMART_PUNCTUATION
}

fn strip_generics_from_path_segment(segment: Vec<char>) -> Result<String, MalformedGenerics> {
    let mut stripped_segment = String::new();
    let mut param_depth = 0;

    let mut latest_generics_chunk = String::new();

    for c in segment {
        if c == '<' {
            param_depth += 1;
            latest_generics_chunk.clear();
        } else if c == '>' {
            param_depth -= 1;
            if latest_generics_chunk.contains(" as ") {
                // The segment tries to use fully-qualified syntax, which is currently unsupported.
                // Give a helpful error message instead of completely ignoring the angle brackets.
                return Err(MalformedGenerics::HasFullyQualifiedSyntax);
            }
        } else if param_depth == 0 {
            stripped_segment.push(c);
        } else {
            latest_generics_chunk.push(c);
        }
    }

    if param_depth == 0 {
        Ok(stripped_segment)
    } else {
        // The segment has unbalanced angle brackets, e.g. `Vec<T` or `Vec<T>>`
        Err(MalformedGenerics::UnbalancedAngleBrackets)
    }
}

pub fn strip_generics_from_path(path_str: &str) -> Result<Box<str>, MalformedGenerics> {
    if !path_str.contains(['<', '>']) {
        return Ok(path_str.into());
    }
    let mut stripped_segments = vec![];
    let mut path = path_str.chars().peekable();
    let mut segment = Vec::new();

    while let Some(chr) = path.next() {
        match chr {
            ':' => {
                if path.next_if_eq(&':').is_some() {
                    let stripped_segment =
                        strip_generics_from_path_segment(mem::take(&mut segment))?;
                    if !stripped_segment.is_empty() {
                        stripped_segments.push(stripped_segment);
                    }
                } else {
                    return Err(MalformedGenerics::InvalidPathSeparator);
                }
            }
            '<' => {
                segment.push(chr);

                match path.next() {
                    Some('<') => {
                        return Err(MalformedGenerics::TooManyAngleBrackets);
                    }
                    Some('>') => {
                        return Err(MalformedGenerics::EmptyAngleBrackets);
                    }
                    Some(chr) => {
                        segment.push(chr);

                        while let Some(chr) = path.next_if(|c| *c != '>') {
                            segment.push(chr);
                        }
                    }
                    None => break,
                }
            }
            _ => segment.push(chr),
        }
        trace!("raw segment: {:?}", segment);
    }

    if !segment.is_empty() {
        let stripped_segment = strip_generics_from_path_segment(segment)?;
        if !stripped_segment.is_empty() {
            stripped_segments.push(stripped_segment);
        }
    }

    debug!("path_str: {path_str:?}\nstripped segments: {stripped_segments:?}");

    let stripped_path = stripped_segments.join("::");

    if !stripped_path.is_empty() {
        Ok(stripped_path.into())
    } else {
        Err(MalformedGenerics::MissingType)
    }
}

/// Returns whether the first doc-comment is an inner attribute.
///
//// If there are no doc-comments, return true.
/// FIXME(#78591): Support both inner and outer attributes on the same item.
pub fn inner_docs(attrs: &[impl AttributeExt]) -> bool {
    attrs
        .iter()
        .find(|a| a.doc_str().is_some())
        .map_or(true, |a| a.style() == ast::AttrStyle::Inner)
}

/// Has `#[rustc_doc_primitive]` or `#[doc(keyword)]`.
pub fn has_primitive_or_keyword_docs(attrs: &[impl AttributeExt]) -> bool {
    for attr in attrs {
        if attr.has_name(sym::rustc_doc_primitive) {
            return true;
        } else if attr.has_name(sym::doc)
            && let Some(items) = attr.meta_item_list()
        {
            for item in items {
                if item.has_name(sym::keyword) {
                    return true;
                }
            }
        }
    }
    false
}

/// Simplified version of the corresponding function in rustdoc.
/// If the rustdoc version returns a successful result, this function must return the same result.
/// Otherwise this function may return anything.
fn preprocess_link(link: &str) -> Box<str> {
    let link = link.replace('`', "");
    let link = link.split('#').next().unwrap();
    let link = link.trim();
    let link = link.rsplit('@').next().unwrap();
    let link = link.strip_suffix("()").unwrap_or(link);
    let link = link.strip_suffix("{}").unwrap_or(link);
    let link = link.strip_suffix("[]").unwrap_or(link);
    let link = if link != "!" { link.strip_suffix('!').unwrap_or(link) } else { link };
    let link = link.trim();
    strip_generics_from_path(link).unwrap_or_else(|_| link.into())
}

/// Keep inline and reference links `[]`,
/// but skip autolinks `<>` which we never consider to be intra-doc links.
pub fn may_be_doc_link(link_type: LinkType) -> bool {
    match link_type {
        LinkType::Inline
        | LinkType::Reference
        | LinkType::ReferenceUnknown
        | LinkType::Collapsed
        | LinkType::CollapsedUnknown
        | LinkType::Shortcut
        | LinkType::ShortcutUnknown => true,
        LinkType::Autolink | LinkType::Email => false,
    }
}

/// Simplified version of `preprocessed_markdown_links` from rustdoc.
/// Must return at least the same links as it, but may add some more links on top of that.
pub(crate) fn attrs_to_preprocessed_links<A: AttributeExt + Clone>(attrs: &[A]) -> Vec<Box<str>> {
    let (doc_fragments, _) = attrs_to_doc_fragments(attrs.iter().map(|attr| (attr, None)), true);
    let doc = prepare_to_doc_link_resolution(&doc_fragments).into_values().next().unwrap();

    parse_links(&doc)
}

/// Similar version of `markdown_links` from rustdoc.
/// This will collect destination links and display text if exists.
fn parse_links<'md>(doc: &'md str) -> Vec<Box<str>> {
    let mut broken_link_callback = |link: BrokenLink<'md>| Some((link.reference, "".into()));
    let mut event_iter = Parser::new_with_broken_link_callback(
        doc,
        main_body_opts(),
        Some(&mut broken_link_callback),
    );
    let mut links = Vec::new();

    while let Some(event) = event_iter.next() {
        match event {
            Event::Start(Tag::Link { link_type, dest_url, title: _, id: _ })
                if may_be_doc_link(link_type) =>
            {
                if matches!(
                    link_type,
                    LinkType::Inline
                        | LinkType::ReferenceUnknown
                        | LinkType::Reference
                        | LinkType::Shortcut
                        | LinkType::ShortcutUnknown
                ) {
                    if let Some(display_text) = collect_link_data(&mut event_iter) {
                        links.push(display_text);
                    }
                }

                links.push(preprocess_link(&dest_url));
            }
            _ => {}
        }
    }

    links
}

/// Collects additional data of link.
fn collect_link_data<'input, F: BrokenLinkCallback<'input>>(
    event_iter: &mut Parser<'input, F>,
) -> Option<Box<str>> {
    let mut display_text: Option<String> = None;
    let mut append_text = |text: CowStr<'_>| {
        if let Some(display_text) = &mut display_text {
            display_text.push_str(&text);
        } else {
            display_text = Some(text.to_string());
        }
    };

    while let Some(event) = event_iter.next() {
        match event {
            Event::Text(text) => {
                append_text(text);
            }
            Event::Code(code) => {
                append_text(code);
            }
            Event::End(_) => {
                break;
            }
            _ => {}
        }
    }

    display_text.map(String::into_boxed_str)
}

/// Returns a span encompassing all the document fragments.
pub fn span_of_fragments(fragments: &[DocFragment]) -> Option<Span> {
    if fragments.is_empty() {
        return None;
    }
    let start = fragments[0].span;
    if start == DUMMY_SP {
        return None;
    }
    let end = fragments.last().expect("no doc strings provided").span;
    Some(start.to(end))
}

/// Attempts to match a range of bytes from parsed markdown to a `Span` in the source code.
///
/// This method does not always work, because markdown bytes don't necessarily match source bytes,
/// like if escapes are used in the string. In this case, it returns `None`.
///
/// This method will return `Some` only if:
///
/// - The doc is made entirely from sugared doc comments, which cannot contain escapes
/// - The doc is entirely from a single doc fragment, with a string literal, exactly equal
/// - The doc comes from `include_str!`
pub fn source_span_for_markdown_range(
    tcx: TyCtxt<'_>,
    markdown: &str,
    md_range: &Range<usize>,
    fragments: &[DocFragment],
) -> Option<Span> {
    if let &[fragment] = &fragments
        && fragment.kind == DocFragmentKind::RawDoc
        && let Ok(snippet) = tcx.sess.source_map().span_to_snippet(fragment.span)
        && snippet.trim_end() == markdown.trim_end()
        && let Ok(md_range_lo) = u32::try_from(md_range.start)
        && let Ok(md_range_hi) = u32::try_from(md_range.end)
    {
        // Single fragment with string that contains same bytes as doc.
        return Some(Span::new(
            fragment.span.lo() + rustc_span::BytePos(md_range_lo),
            fragment.span.lo() + rustc_span::BytePos(md_range_hi),
            fragment.span.ctxt(),
            fragment.span.parent(),
        ));
    }

    let is_all_sugared_doc = fragments.iter().all(|frag| frag.kind == DocFragmentKind::SugaredDoc);

    if !is_all_sugared_doc {
        return None;
    }

    let snippet = tcx.sess.source_map().span_to_snippet(span_of_fragments(fragments)?).ok()?;

    let starting_line = markdown[..md_range.start].matches('\n').count();
    let ending_line = starting_line + markdown[md_range.start..md_range.end].matches('\n').count();

    // We use `split_terminator('\n')` instead of `lines()` when counting bytes so that we treat
    // CRLF and LF line endings the same way.
    let mut src_lines = snippet.split_terminator('\n');
    let md_lines = markdown.split_terminator('\n');

    // The number of bytes from the source span to the markdown span that are not part
    // of the markdown, like comment markers.
    let mut start_bytes = 0;
    let mut end_bytes = 0;

    'outer: for (line_no, md_line) in md_lines.enumerate() {
        loop {
            let source_line = src_lines.next()?;
            match source_line.find(md_line) {
                Some(offset) => {
                    if line_no == starting_line {
                        start_bytes += offset;

                        if starting_line == ending_line {
                            break 'outer;
                        }
                    } else if line_no == ending_line {
                        end_bytes += offset;
                        break 'outer;
                    } else if line_no < starting_line {
                        start_bytes += source_line.len() - md_line.len();
                    } else {
                        end_bytes += source_line.len() - md_line.len();
                    }
                    break;
                }
                None => {
                    // Since this is a source line that doesn't include a markdown line,
                    // we have to count the newline that we split from earlier.
                    if line_no <= starting_line {
                        start_bytes += source_line.len() + 1;
                    } else {
                        end_bytes += source_line.len() + 1;
                    }
                }
            }
        }
    }

    Some(span_of_fragments(fragments)?.from_inner(InnerSpan::new(
        md_range.start + start_bytes,
        md_range.end + start_bytes + end_bytes,
    )))
}
