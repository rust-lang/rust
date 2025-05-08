//! Extracts, resolves and rewrites links and intra-doc links in markdown documentation.

#[cfg(test)]
mod tests;

mod intra_doc_links;

use std::ops::Range;

use pulldown_cmark::{BrokenLink, CowStr, Event, InlineStr, LinkType, Options, Parser, Tag};
use pulldown_cmark_to_cmark::{Options as CMarkOptions, cmark_resume_with_options};
use stdx::format_to;
use url::Url;

use hir::{
    Adt, AsAssocItem, AssocItem, AssocItemContainer, AttrsWithOwner, HasAttrs, db::HirDatabase, sym,
};
use ide_db::{
    RootDatabase,
    base_db::{CrateOrigin, LangCrateOrigin, ReleaseChannel, RootQueryDb},
    defs::{Definition, NameClass, NameRefClass},
    documentation::{DocsRangeMap, Documentation, HasDocs, docs_with_rangemap},
    helpers::pick_best_token,
};
use syntax::{
    AstNode, AstToken,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken, T, TextRange, TextSize,
    ast::{self, IsString},
    match_ast,
};

use crate::{
    FilePosition, Semantics,
    doc_links::intra_doc_links::{parse_intra_doc_link, strip_prefixes_suffixes},
};

/// Web and local links to an item's documentation.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct DocumentationLinks {
    /// The URL to the documentation on docs.rs.
    /// May not lead anywhere.
    pub web_url: Option<String>,
    /// The URL to the documentation in the local file system.
    /// May not lead anywhere.
    pub local_url: Option<String>,
}

const MARKDOWN_OPTIONS: Options =
    Options::ENABLE_FOOTNOTES.union(Options::ENABLE_TABLES).union(Options::ENABLE_TASKLISTS);

/// Rewrite documentation links in markdown to point to an online host (e.g. docs.rs)
pub(crate) fn rewrite_links(
    db: &RootDatabase,
    markdown: &str,
    definition: Definition,
    range_map: Option<DocsRangeMap>,
) -> String {
    let mut cb = broken_link_clone_cb;
    let doc = Parser::new_with_broken_link_callback(markdown, MARKDOWN_OPTIONS, Some(&mut cb))
        .into_offset_iter();

    let doc = map_links(doc, |target, title, range| {
        // This check is imperfect, there's some overlap between valid intra-doc links
        // and valid URLs so we choose to be too eager to try to resolve what might be
        // a URL.
        if target.contains("://") {
            (Some(LinkType::Inline), target.to_owned(), title.to_owned())
        } else {
            // Two possibilities:
            // * path-based links: `../../module/struct.MyStruct.html`
            // * module-based links (AKA intra-doc links): `super::super::module::MyStruct`
            let text_range =
                TextRange::new(range.start.try_into().unwrap(), range.end.try_into().unwrap());
            let is_inner_doc = range_map
                .as_ref()
                .and_then(|range_map| range_map.map(text_range))
                .map(|(_, attr_id)| attr_id.is_inner_attr())
                .unwrap_or(false);
            if let Some((target, title)) =
                rewrite_intra_doc_link(db, definition, target, title, is_inner_doc)
            {
                (None, target, title)
            } else if let Some(target) = rewrite_url_link(db, definition, target) {
                (Some(LinkType::Inline), target, title.to_owned())
            } else {
                (None, target.to_owned(), title.to_owned())
            }
        }
    });
    let mut out = String::new();
    cmark_resume_with_options(
        doc,
        &mut out,
        None,
        CMarkOptions { code_block_token_count: 3, ..Default::default() },
    )
    .ok();
    out
}

/// Remove all links in markdown documentation.
pub(crate) fn remove_links(markdown: &str) -> String {
    let mut drop_link = false;

    let mut cb = |_: BrokenLink<'_>| {
        let empty = InlineStr::try_from("").unwrap();
        Some((CowStr::Inlined(empty), CowStr::Inlined(empty)))
    };
    let doc = Parser::new_with_broken_link_callback(markdown, MARKDOWN_OPTIONS, Some(&mut cb));
    let doc = doc.filter_map(move |evt| match evt {
        Event::Start(Tag::Link(link_type, target, title)) => {
            if link_type == LinkType::Inline && target.contains("://") {
                Some(Event::Start(Tag::Link(link_type, target, title)))
            } else {
                drop_link = true;
                None
            }
        }
        Event::End(_) if drop_link => {
            drop_link = false;
            None
        }
        _ => Some(evt),
    });

    let mut out = String::new();
    cmark_resume_with_options(
        doc,
        &mut out,
        None,
        CMarkOptions { code_block_token_count: 3, ..Default::default() },
    )
    .ok();
    out
}

// Feature: Open Docs
//
// Retrieve a links to documentation for the given symbol.
//
// The simplest way to use this feature is via the context menu. Right-click on
// the selected item. The context menu opens. Select **Open Docs**.
//
// | Editor  | Action Name |
// |---------|-------------|
// | VS Code | **rust-analyzer: Open Docs** |
pub(crate) fn external_docs(
    db: &RootDatabase,
    FilePosition { file_id, offset }: FilePosition,
    target_dir: Option<&str>,
    sysroot: Option<&str>,
) -> Option<DocumentationLinks> {
    let sema = &Semantics::new(db);
    let file = sema.parse_guess_edition(file_id).syntax().clone();
    let token = pick_best_token(file.token_at_offset(offset), |kind| match kind {
        IDENT | INT_NUMBER | T![self] => 3,
        T!['('] | T![')'] => 2,
        kind if kind.is_trivia() => 0,
        _ => 1,
    })?;
    let token = sema.descend_into_macros_single_exact(token);

    let node = token.parent()?;
    let definition = match_ast! {
        match node {
            ast::NameRef(name_ref) => match NameRefClass::classify(sema, &name_ref)? {
                NameRefClass::Definition(def, _) => def,
                NameRefClass::FieldShorthand { local_ref: _, field_ref, adt_subst: _ } => {
                    Definition::Field(field_ref)
                }
                NameRefClass::ExternCrateShorthand { decl, .. } => {
                    Definition::ExternCrateDecl(decl)
                }
            },
            ast::Name(name) => match NameClass::classify(sema, &name)? {
                NameClass::Definition(it) | NameClass::ConstReference(it) => it,
                NameClass::PatFieldShorthand { local_def: _, field_ref, adt_subst: _ } => Definition::Field(field_ref),
            },
            _ => return None
        }
    };

    Some(get_doc_links(db, definition, target_dir, sysroot))
}

/// Extracts all links from a given markdown text returning the definition text range, link-text
/// and the namespace if known.
pub(crate) fn extract_definitions_from_docs(
    docs: &Documentation,
) -> Vec<(TextRange, String, Option<hir::Namespace>)> {
    Parser::new_with_broken_link_callback(
        docs.as_str(),
        MARKDOWN_OPTIONS,
        Some(&mut broken_link_clone_cb),
    )
    .into_offset_iter()
    .filter_map(|(event, range)| match event {
        Event::Start(Tag::Link(_, target, _)) => {
            let (link, ns) = parse_intra_doc_link(&target);
            Some((
                TextRange::new(range.start.try_into().ok()?, range.end.try_into().ok()?),
                link.to_owned(),
                ns,
            ))
        }
        _ => None,
    })
    .collect()
}

pub(crate) fn resolve_doc_path_for_def(
    db: &dyn HirDatabase,
    def: Definition,
    link: &str,
    ns: Option<hir::Namespace>,
    is_inner_doc: bool,
) -> Option<Definition> {
    match def {
        Definition::Module(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::Crate(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::Function(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::Adt(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::Variant(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::Const(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::Static(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::Trait(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::TraitAlias(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::TypeAlias(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::Macro(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::Field(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::SelfType(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::ExternCrateDecl(it) => it.resolve_doc_path(db, link, ns, is_inner_doc),
        Definition::BuiltinAttr(_)
        | Definition::BuiltinType(_)
        | Definition::BuiltinLifetime(_)
        | Definition::ToolModule(_)
        | Definition::TupleField(_)
        | Definition::Local(_)
        | Definition::GenericParam(_)
        | Definition::Label(_)
        | Definition::DeriveHelper(_)
        | Definition::InlineAsmRegOrRegClass(_)
        | Definition::InlineAsmOperand(_) => None,
    }
    .map(Definition::from)
}

pub(crate) fn doc_attributes(
    sema: &Semantics<'_, RootDatabase>,
    node: &SyntaxNode,
) -> Option<(hir::AttrsWithOwner, Definition)> {
    match_ast! {
        match node {
            ast::SourceFile(it)  => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            ast::Module(it)      => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            ast::Fn(it)          => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            ast::Struct(it)      => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(hir::Adt::Struct(def)))),
            ast::Union(it)       => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(hir::Adt::Union(def)))),
            ast::Enum(it)        => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(hir::Adt::Enum(def)))),
            ast::Variant(it)     => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            ast::Trait(it)       => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            ast::Static(it)      => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            ast::Const(it)       => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            ast::TypeAlias(it)   => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            ast::Impl(it)        => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            ast::RecordField(it) => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            ast::TupleField(it)  => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            ast::Macro(it)       => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            ast::ExternCrate(it) => sema.to_def(&it).map(|def| (def.attrs(sema.db), Definition::from(def))),
            // ast::Use(it) => sema.to_def(&it).map(|def| (Box::new(it) as _, def.attrs(sema.db))),
            _ => None
        }
    }
}

pub(crate) struct DocCommentToken {
    doc_token: SyntaxToken,
    prefix_len: TextSize,
}

pub(crate) fn token_as_doc_comment(doc_token: &SyntaxToken) -> Option<DocCommentToken> {
    (match_ast! {
        match doc_token {
            ast::Comment(comment) => TextSize::try_from(comment.prefix().len()).ok(),
            ast::String(string) => {
                doc_token.parent_ancestors().find_map(ast::Attr::cast).filter(|attr| attr.simple_name().as_deref() == Some("doc"))?;
                if doc_token.parent_ancestors().find_map(ast::MacroCall::cast).filter(|mac| mac.path().and_then(|p| p.segment()?.name_ref()).as_ref().map(|n| n.text()).as_deref() == Some("include_str")).is_some() {
                    return None;
                }
                string.open_quote_text_range().map(|it| it.len())
            },
            _ => None,
        }
    }).map(|prefix_len| DocCommentToken { prefix_len, doc_token: doc_token.clone() })
}

impl DocCommentToken {
    pub(crate) fn get_definition_with_descend_at<T>(
        self,
        sema: &Semantics<'_, RootDatabase>,
        offset: TextSize,
        // Definition, CommentOwner, range of intra doc link in original file
        mut cb: impl FnMut(Definition, SyntaxNode, TextRange) -> Option<T>,
    ) -> Option<T> {
        let DocCommentToken { prefix_len, doc_token } = self;
        // offset relative to the comments contents
        let original_start = doc_token.text_range().start();
        let relative_comment_offset = offset - original_start - prefix_len;

        sema.descend_into_macros(doc_token).into_iter().find_map(|t| {
            let (node, descended_prefix_len, is_inner) = match_ast!{
                match t {
                    ast::Comment(comment) => {
                        (t.parent()?, TextSize::try_from(comment.prefix().len()).ok()?, comment.is_inner())
                    },
                    ast::String(string) => {
                        let attr = t.parent_ancestors().find_map(ast::Attr::cast)?;
                        let attr_is_inner = attr.excl_token().map(|excl| excl.kind() == BANG).unwrap_or(false);
                        (attr.syntax().parent()?, string.open_quote_text_range()?.len(), attr_is_inner)
                    },
                    _ => return None,
                }
            };
            let token_start = t.text_range().start();
            let abs_in_expansion_offset = token_start + relative_comment_offset + descended_prefix_len;
            let (attributes, def) = Self::doc_attributes(sema, &node, is_inner)?;
            let (docs, doc_mapping) = docs_with_rangemap(sema.db, &attributes)?;
            let (in_expansion_range, link, ns, is_inner) =
                extract_definitions_from_docs(&docs).into_iter().find_map(|(range, link, ns)| {
                    let (mapped, idx) = doc_mapping.map(range)?;
                    (mapped.value.contains(abs_in_expansion_offset)).then_some((mapped.value, link, ns, idx.is_inner_attr()))
                })?;
            // get the relative range to the doc/attribute in the expansion
            let in_expansion_relative_range = in_expansion_range - descended_prefix_len - token_start;
            // Apply relative range to the original input comment
            let absolute_range = in_expansion_relative_range + original_start + prefix_len;
            let def = resolve_doc_path_for_def(sema.db, def, &link, ns, is_inner)?;
            cb(def, node, absolute_range)
        })
    }

    /// When we hover a inner doc item, this find a attached definition.
    /// ```
    /// // node == ITEM_LIST
    /// // node.parent == EXPR_BLOCK
    /// // node.parent().parent() == FN
    /// fn f() {
    ///    //! [`S$0`]
    /// }
    /// ```
    fn doc_attributes(
        sema: &Semantics<'_, RootDatabase>,
        node: &SyntaxNode,
        is_inner_doc: bool,
    ) -> Option<(AttrsWithOwner, Definition)> {
        if is_inner_doc && node.kind() != SOURCE_FILE {
            let parent = node.parent()?;
            doc_attributes(sema, &parent).or(doc_attributes(sema, &parent.parent()?))
        } else {
            doc_attributes(sema, node)
        }
    }
}

fn broken_link_clone_cb(link: BrokenLink<'_>) -> Option<(CowStr<'_>, CowStr<'_>)> {
    Some((/*url*/ link.reference.clone(), /*title*/ link.reference))
}

// FIXME:
// BUG: For Option::Some
// Returns https://doc.rust-lang.org/nightly/core/prelude/v1/enum.Option.html#variant.Some
// Instead of https://doc.rust-lang.org/nightly/core/option/enum.Option.html
//
// This should cease to be a problem if RFC2988 (Stable Rustdoc URLs) is implemented
// https://github.com/rust-lang/rfcs/pull/2988
fn get_doc_links(
    db: &RootDatabase,
    def: Definition,
    target_dir: Option<&str>,
    sysroot: Option<&str>,
) -> DocumentationLinks {
    let join_url = |base_url: Option<Url>, path: &str| -> Option<Url> {
        base_url.and_then(|url| url.join(path).ok())
    };

    let Some((target, file, frag)) = filename_and_frag_for_def(db, def) else {
        return Default::default();
    };

    let (mut web_url, mut local_url) = get_doc_base_urls(db, target, target_dir, sysroot);

    if let Some(path) = mod_path_of_def(db, target) {
        web_url = join_url(web_url, &path);
        local_url = join_url(local_url, &path);
    }

    web_url = join_url(web_url, &file);
    local_url = join_url(local_url, &file);

    if let Some(url) = web_url.as_mut() {
        url.set_fragment(frag.as_deref())
    }
    if let Some(url) = local_url.as_mut() {
        url.set_fragment(frag.as_deref())
    }

    DocumentationLinks {
        web_url: web_url.map(|it| it.into()),
        local_url: local_url.map(|it| it.into()),
    }
}

fn rewrite_intra_doc_link(
    db: &RootDatabase,
    def: Definition,
    target: &str,
    title: &str,
    is_inner_doc: bool,
) -> Option<(String, String)> {
    let (link, ns) = parse_intra_doc_link(target);

    let (link, anchor) = match link.split_once('#') {
        Some((new_link, anchor)) => (new_link, Some(anchor)),
        None => (link, None),
    };

    let resolved = resolve_doc_path_for_def(db, def, link, ns, is_inner_doc)?;
    let mut url = get_doc_base_urls(db, resolved, None, None).0?;

    let (_, file, frag) = filename_and_frag_for_def(db, resolved)?;
    if let Some(path) = mod_path_of_def(db, resolved) {
        url = url.join(&path).ok()?;
    }

    let frag = anchor.or(frag.as_deref());

    url = url.join(&file).ok()?;
    url.set_fragment(frag);

    Some((url.into(), strip_prefixes_suffixes(title).to_owned()))
}

/// Try to resolve path to local documentation via path-based links (i.e. `../gateway/struct.Shard.html`).
fn rewrite_url_link(db: &RootDatabase, def: Definition, target: &str) -> Option<String> {
    if !(target.contains('#') || target.contains(".html")) {
        return None;
    }

    let mut url = get_doc_base_urls(db, def, None, None).0?;
    let (def, file, frag) = filename_and_frag_for_def(db, def)?;

    if let Some(path) = mod_path_of_def(db, def) {
        url = url.join(&path).ok()?;
    }

    url = url.join(&file).ok()?;
    url.set_fragment(frag.as_deref());
    url.join(target).ok().map(Into::into)
}

fn mod_path_of_def(db: &RootDatabase, def: Definition) -> Option<String> {
    def.canonical_module_path(db).map(|it| {
        let mut path = String::new();
        it.flat_map(|it| it.name(db)).for_each(|name| format_to!(path, "{}/", name.as_str()));
        path
    })
}

/// Rewrites a markdown document, applying 'callback' to each link.
fn map_links<'e>(
    events: impl Iterator<Item = (Event<'e>, Range<usize>)>,
    callback: impl Fn(&str, &str, Range<usize>) -> (Option<LinkType>, String, String),
) -> impl Iterator<Item = Event<'e>> {
    let mut in_link = false;
    // holds the origin link target on start event and the rewritten one on end event
    let mut end_link_target: Option<CowStr<'_>> = None;
    // normally link's type is determined by the type of link tag in the end event,
    // however in some cases we want to change the link type, for example,
    // `Shortcut` type parsed from Start/End tags doesn't make sense for url links
    let mut end_link_type: Option<LinkType> = None;

    events.map(move |(evt, range)| match evt {
        Event::Start(Tag::Link(link_type, ref target, _)) => {
            in_link = true;
            end_link_target = Some(target.clone());
            end_link_type = Some(link_type);
            evt
        }
        Event::End(Tag::Link(link_type, target, _)) => {
            in_link = false;
            Event::End(Tag::Link(
                end_link_type.unwrap_or(link_type),
                end_link_target.take().unwrap_or(target),
                CowStr::Borrowed(""),
            ))
        }
        Event::Text(s) if in_link => {
            let (link_type, link_target_s, link_name) =
                callback(&end_link_target.take().unwrap(), &s, range);
            end_link_target = Some(CowStr::Boxed(link_target_s.into()));
            if !matches!(end_link_type, Some(LinkType::Autolink)) {
                end_link_type = link_type;
            }
            Event::Text(CowStr::Boxed(link_name.into()))
        }
        Event::Code(s) if in_link => {
            let (link_type, link_target_s, link_name) =
                callback(&end_link_target.take().unwrap(), &s, range);
            end_link_target = Some(CowStr::Boxed(link_target_s.into()));
            if !matches!(end_link_type, Some(LinkType::Autolink)) {
                end_link_type = link_type;
            }
            Event::Code(CowStr::Boxed(link_name.into()))
        }
        _ => evt,
    })
}

/// Get the root URL for the documentation of a definition.
///
/// ```ignore
/// https://doc.rust-lang.org/std/iter/trait.Iterator.html#tymethod.next
/// ^^^^^^^^^^^^^^^^^^^^^^^^^^
/// file:///project/root/target/doc/std/iter/trait.Iterator.html#tymethod.next
/// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/// ```
fn get_doc_base_urls(
    db: &RootDatabase,
    def: Definition,
    target_dir: Option<&str>,
    sysroot: Option<&str>,
) -> (Option<Url>, Option<Url>) {
    let local_doc = target_dir
        .and_then(|path| Url::parse(&format!("file:///{path}/")).ok())
        .and_then(|it| it.join("doc/").ok());
    let system_doc = sysroot
        .map(|sysroot| format!("file:///{sysroot}/share/doc/rust/html/"))
        .and_then(|it| Url::parse(&it).ok());
    let krate = def.krate(db);
    let channel = krate
        .and_then(|krate| db.toolchain_channel(krate.into()))
        .unwrap_or(ReleaseChannel::Nightly)
        .as_str();

    // special case base url of `BuiltinType` to core
    // https://github.com/rust-lang/rust-analyzer/issues/12250
    if let Definition::BuiltinType(..) = def {
        let web_link = Url::parse(&format!("https://doc.rust-lang.org/{channel}/core/")).ok();
        let system_link = system_doc.and_then(|it| it.join("core/").ok());
        return (web_link, system_link);
    };

    let Some(krate) = krate else { return Default::default() };
    let Some(display_name) = krate.display_name(db) else { return Default::default() };
    let (web_base, local_base) = match krate.origin(db) {
        // std and co do not specify `html_root_url` any longer so we gotta handwrite this ourself.
        // FIXME: Use the toolchains channel instead of nightly
        CrateOrigin::Lang(
            origin @ (LangCrateOrigin::Alloc
            | LangCrateOrigin::Core
            | LangCrateOrigin::ProcMacro
            | LangCrateOrigin::Std
            | LangCrateOrigin::Test),
        ) => {
            let system_url = system_doc.and_then(|it| it.join(&format!("{origin}")).ok());
            let web_url = format!("https://doc.rust-lang.org/{channel}/{origin}");
            (Some(web_url), system_url)
        }
        CrateOrigin::Lang(_) => return (None, None),
        CrateOrigin::Rustc { name: _ } => {
            (Some(format!("https://doc.rust-lang.org/{channel}/nightly-rustc/")), None)
        }
        CrateOrigin::Local { repo: _, name: _ } => {
            // FIXME: These should not attempt to link to docs.rs!
            let weblink = krate.get_html_root_url(db).or_else(|| {
                let version = krate.version(db);
                // Fallback to docs.rs. This uses `display_name` and can never be
                // correct, but that's what fallbacks are about.
                //
                // FIXME: clicking on the link should just open the file in the editor,
                // instead of falling back to external urls.
                Some(format!(
                    "https://docs.rs/{krate}/{version}/",
                    krate = display_name,
                    version = version.as_deref().unwrap_or("*")
                ))
            });
            (weblink, local_doc)
        }
        CrateOrigin::Library { repo: _, name } => {
            let weblink = krate.get_html_root_url(db).or_else(|| {
                let version = krate.version(db);
                // Fallback to docs.rs. This uses `display_name` and can never be
                // correct, but that's what fallbacks are about.
                //
                // FIXME: clicking on the link should just open the file in the editor,
                // instead of falling back to external urls.
                Some(format!(
                    "https://docs.rs/{krate}/{version}/",
                    krate = name,
                    version = version.as_deref().unwrap_or("*")
                ))
            });
            (weblink, local_doc)
        }
    };
    let web_base = web_base
        .and_then(|it| Url::parse(&it).ok())
        .and_then(|it| it.join(&format!("{display_name}/")).ok());
    let local_base = local_base.and_then(|it| it.join(&format!("{display_name}/")).ok());

    (web_base, local_base)
}

/// Get the filename and extension generated for a symbol by rustdoc.
///
/// ```ignore
/// https://doc.rust-lang.org/std/iter/trait.Iterator.html#tymethod.next
///                                    ^^^^^^^^^^^^^^^^^^^
/// ```
fn filename_and_frag_for_def(
    db: &dyn HirDatabase,
    def: Definition,
) -> Option<(Definition, String, Option<String>)> {
    if let Some(assoc_item) = def.as_assoc_item(db) {
        let def = match assoc_item.container(db) {
            AssocItemContainer::Trait(t) => t.into(),
            AssocItemContainer::Impl(i) => i.self_ty(db).as_adt()?.into(),
        };
        let (_, file, _) = filename_and_frag_for_def(db, def)?;
        let frag = get_assoc_item_fragment(db, assoc_item)?;
        return Some((def, file, Some(frag)));
    }

    let res = match def {
        Definition::Adt(adt) => match adt {
            Adt::Struct(s) => {
                format!("struct.{}.html", s.name(db).as_str())
            }
            Adt::Enum(e) => format!("enum.{}.html", e.name(db).as_str()),
            Adt::Union(u) => format!("union.{}.html", u.name(db).as_str()),
        },
        Definition::Crate(_) => String::from("index.html"),
        Definition::Module(m) => match m.name(db) {
            // `#[doc(keyword = "...")]` is internal used only by rust compiler
            Some(name) => {
                match m.attrs(db).by_key(sym::doc).find_string_value_in_tt(sym::keyword) {
                    Some(kw) => {
                        format!("keyword.{kw}.html")
                    }
                    None => format!("{}/index.html", name.as_str()),
                }
            }
            None => String::from("index.html"),
        },
        Definition::Trait(t) => {
            format!("trait.{}.html", t.name(db).as_str())
        }
        Definition::TraitAlias(t) => {
            format!("traitalias.{}.html", t.name(db).as_str())
        }
        Definition::TypeAlias(t) => {
            format!("type.{}.html", t.name(db).as_str())
        }
        Definition::BuiltinType(t) => {
            format!("primitive.{}.html", t.name().as_str())
        }
        Definition::Function(f) => {
            format!("fn.{}.html", f.name(db).as_str())
        }
        Definition::Variant(ev) => {
            let def = Definition::Adt(ev.parent_enum(db).into());
            let (_, file, _) = filename_and_frag_for_def(db, def)?;
            return Some((def, file, Some(format!("variant.{}", ev.name(db).as_str()))));
        }
        Definition::Const(c) => {
            format!("constant.{}.html", c.name(db)?.as_str())
        }
        Definition::Static(s) => {
            format!("static.{}.html", s.name(db).as_str())
        }
        Definition::Macro(mac) => match mac.kind(db) {
            hir::MacroKind::Declarative
            | hir::MacroKind::AttrBuiltIn
            | hir::MacroKind::DeclarativeBuiltIn
            | hir::MacroKind::Attr
            | hir::MacroKind::ProcMacro => {
                format!("macro.{}.html", mac.name(db).as_str())
            }
            hir::MacroKind::Derive | hir::MacroKind::DeriveBuiltIn => {
                format!("derive.{}.html", mac.name(db).as_str())
            }
        },
        Definition::Field(field) => {
            let def = match field.parent_def(db) {
                hir::VariantDef::Struct(it) => Definition::Adt(it.into()),
                hir::VariantDef::Union(it) => Definition::Adt(it.into()),
                hir::VariantDef::Variant(it) => Definition::Variant(it),
            };
            let (_, file, _) = filename_and_frag_for_def(db, def)?;
            return Some((def, file, Some(format!("structfield.{}", field.name(db).as_str()))));
        }
        Definition::SelfType(impl_) => {
            let adt = impl_.self_ty(db).as_adt()?.into();
            let (_, file, _) = filename_and_frag_for_def(db, adt)?;
            // FIXME fragment numbering
            return Some((adt, file, Some(String::from("impl"))));
        }
        Definition::ExternCrateDecl(it) => {
            format!("{}/index.html", it.name(db).as_str())
        }
        Definition::Local(_)
        | Definition::GenericParam(_)
        | Definition::TupleField(_)
        | Definition::Label(_)
        | Definition::BuiltinAttr(_)
        | Definition::BuiltinLifetime(_)
        | Definition::ToolModule(_)
        | Definition::DeriveHelper(_)
        | Definition::InlineAsmRegOrRegClass(_)
        | Definition::InlineAsmOperand(_) => return None,
    };

    Some((def, res, None))
}

/// Get the fragment required to link to a specific field, method, associated type, or associated constant.
///
/// ```ignore
/// https://doc.rust-lang.org/std/iter/trait.Iterator.html#tymethod.next
///                                                       ^^^^^^^^^^^^^^
/// ```
fn get_assoc_item_fragment(db: &dyn HirDatabase, assoc_item: hir::AssocItem) -> Option<String> {
    Some(match assoc_item {
        AssocItem::Function(function) => {
            let is_trait_method =
                function.as_assoc_item(db).and_then(|assoc| assoc.container_trait(db)).is_some();
            // This distinction may get more complicated when specialization is available.
            // Rustdoc makes this decision based on whether a method 'has defaultness'.
            // Currently this is only the case for provided trait methods.
            if is_trait_method && !function.has_body(db) {
                format!("tymethod.{}", function.name(db).as_str())
            } else {
                format!("method.{}", function.name(db).as_str())
            }
        }
        AssocItem::Const(constant) => {
            format!("associatedconstant.{}", constant.name(db)?.as_str())
        }
        AssocItem::TypeAlias(ty) => {
            format!("associatedtype.{}", ty.name(db).as_str())
        }
    })
}
