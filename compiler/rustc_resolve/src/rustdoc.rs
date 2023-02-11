use pulldown_cmark::{BrokenLink, Event, LinkType, Options, Parser, Tag};
use rustc_ast as ast;
use rustc_ast::util::comments::beautify_doc_string;
use rustc_data_structures::fx::FxHashMap;
use rustc_span::def_id::DefId;
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;
use std::{cmp, mem};

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
            fragment.doc.as_str().lines().fold(usize::MAX, |min_indent, line| {
                if line.chars().all(|c| c.is_whitespace()) {
                    min_indent
                } else {
                    // Compare against either space or tab, ignoring whether they are
                    // mixed or not.
                    let whitespace = line.chars().take_while(|c| *c == ' ' || *c == '\t').count();
                    cmp::min(min_indent, whitespace)
                        + if fragment.kind == DocFragmentKind::SugaredDoc { 0 } else { add }
                }
            })
        })
        .min()
    else {
        return;
    };

    for fragment in docs {
        if fragment.doc == kw::Empty {
            continue;
        }

        let min_indent = if fragment.kind != DocFragmentKind::SugaredDoc && min_indent > 0 {
            min_indent - add
        } else {
            min_indent
        };

        fragment.indent = min_indent;
    }
}

/// The goal of this function is to apply the `DocFragment` transformation that is required when
/// transforming into the final Markdown, which is applying the computed indent to each line in
/// each doc fragment (a `DocFragment` can contain multiple lines in case of `#[doc = ""]`).
///
/// Note: remove the trailing newline where appropriate
pub fn add_doc_fragment(out: &mut String, frag: &DocFragment) {
    let s = frag.doc.as_str();
    let mut iter = s.lines();
    if s.is_empty() {
        out.push('\n');
        return;
    }
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

pub fn attrs_to_doc_fragments<'a>(
    attrs: impl Iterator<Item = (&'a ast::Attribute, Option<DefId>)>,
    doc_only: bool,
) -> (Vec<DocFragment>, ast::AttrVec) {
    let mut doc_fragments = Vec::new();
    let mut other_attrs = ast::AttrVec::new();
    for (attr, item_id) in attrs {
        if let Some((doc_str, comment_kind)) = attr.doc_str_and_comment_kind() {
            let doc = beautify_doc_string(doc_str, comment_kind);
            let kind = if attr.is_doc_comment() {
                DocFragmentKind::SugaredDoc
            } else {
                DocFragmentKind::RawDoc
            };
            let fragment = DocFragment { span: attr.span, doc, kind, item_id, indent: 0 };
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
) -> FxHashMap<Option<DefId>, String> {
    let mut res = FxHashMap::default();
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
        } else {
            if param_depth == 0 {
                stripped_segment.push(c);
            } else {
                latest_generics_chunk.push(c);
            }
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

    debug!("path_str: {:?}\nstripped segments: {:?}", path_str, &stripped_segments);

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
pub fn inner_docs(attrs: &[ast::Attribute]) -> bool {
    attrs.iter().find(|a| a.doc_str().is_some()).map_or(true, |a| a.style == ast::AttrStyle::Inner)
}

/// Has `#[doc(primitive)]` or `#[doc(keyword)]`.
pub fn has_primitive_or_keyword_docs(attrs: &[ast::Attribute]) -> bool {
    for attr in attrs {
        if attr.has_name(sym::doc) && let Some(items) = attr.meta_item_list() {
            for item in items {
                if item.has_name(sym::primitive) || item.has_name(sym::keyword) {
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
pub(crate) fn attrs_to_preprocessed_links(attrs: &[ast::Attribute]) -> Vec<Box<str>> {
    let (doc_fragments, _) = attrs_to_doc_fragments(attrs.iter().map(|attr| (attr, None)), true);
    let doc = prepare_to_doc_link_resolution(&doc_fragments).into_values().next().unwrap();

    Parser::new_with_broken_link_callback(
        &doc,
        main_body_opts(),
        Some(&mut |link: BrokenLink<'_>| Some((link.reference, "".into()))),
    )
    .filter_map(|event| match event {
        Event::Start(Tag::Link(link_type, dest, _)) if may_be_doc_link(link_type) => {
            Some(preprocess_link(&dest))
        }
        _ => None,
    })
    .collect()
}
