use crate::edition::Edition;
use crate::ext::base::{DummyResult, ExtCtxt, MacResult, TTMacroExpander};
use crate::ext::base::{SyntaxExtension, SyntaxExtensionKind};
use crate::ext::expand::{AstFragment, AstFragmentKind};
use crate::ext::hygiene::Transparency;
use crate::ext::tt::macro_parser::{parse, parse_failure_msg};
use crate::ext::tt::macro_parser::{Error, Failure, Success};
use crate::ext::tt::macro_parser::{MatchedNonterminal, MatchedSeq};
use crate::ext::tt::quoted;
use crate::ext::tt::transcribe::transcribe;
use crate::feature_gate::Features;
use crate::parse::parser::Parser;
use crate::parse::token::TokenKind::*;
use crate::parse::token::{self, NtTT, Token};
use crate::parse::{Directory, ParseSess};
use crate::symbol::{kw, sym, Symbol};
use crate::tokenstream::{DelimSpan, TokenStream, TokenTree};
use crate::{ast, attr};

use errors::FatalError;
use log::debug;
use syntax_pos::{symbol::Ident, Span};

use rustc_data_structures::fx::FxHashMap;
use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::slice;

use errors::Applicability;
use rustc_data_structures::sync::Lrc;

const VALID_FRAGMENT_NAMES_MSG: &str = "valid fragment specifiers are \
                                        `ident`, `block`, `stmt`, `expr`, `pat`, `ty`, `lifetime`, \
                                        `literal`, `path`, `meta`, `tt`, `item` and `vis`";

pub struct ParserAnyMacro<'a> {
    parser: Parser<'a>,

    /// Span of the expansion site of the macro this parser is for
    site_span: Span,
    /// The ident of the macro we're parsing
    macro_ident: ast::Ident,
    arm_span: Span,
}

impl<'a> ParserAnyMacro<'a> {
    pub fn make(mut self: Box<ParserAnyMacro<'a>>, kind: AstFragmentKind) -> AstFragment {
        let ParserAnyMacro { site_span, macro_ident, ref mut parser, arm_span } = *self;
        let fragment = panictry!(parser.parse_ast_fragment(kind, true).map_err(|mut e| {
            if parser.token == token::Eof && e.message().ends_with(", found `<eof>`") {
                if !e.span.is_dummy() {
                    // early end of macro arm (#52866)
                    e.replace_span_with(parser.sess.source_map().next_point(parser.token.span));
                }
                let msg = &e.message[0];
                e.message[0] = (
                    format!(
                        "macro expansion ends with an incomplete expression: {}",
                        msg.0.replace(", found `<eof>`", ""),
                    ),
                    msg.1,
                );
            }
            if e.span.is_dummy() {
                // Get around lack of span in error (#30128)
                e.replace_span_with(site_span);
                if parser.sess.source_map().span_to_filename(arm_span).is_real() {
                    e.span_label(arm_span, "in this macro arm");
                }
            } else if !parser.sess.source_map().span_to_filename(parser.token.span).is_real() {
                e.span_label(site_span, "in this macro invocation");
            }
            e
        }));

        // We allow semicolons at the end of expressions -- e.g., the semicolon in
        // `macro_rules! m { () => { panic!(); } }` isn't parsed by `.parse_expr()`,
        // but `m!()` is allowed in expression positions (cf. issue #34706).
        if kind == AstFragmentKind::Expr && parser.token == token::Semi {
            parser.bump();
        }

        // Make sure we don't have any tokens left to parse so we don't silently drop anything.
        let path = ast::Path::from_ident(macro_ident.with_span_pos(site_span));
        parser.ensure_complete_parse(&path, kind.name(), site_span);
        fragment
    }
}

struct MacroRulesMacroExpander {
    name: ast::Ident,
    lhses: Vec<quoted::TokenTree>,
    rhses: Vec<quoted::TokenTree>,
    valid: bool,
}

impl TTMacroExpander for MacroRulesMacroExpander {
    fn expand<'cx>(
        &self,
        cx: &'cx mut ExtCtxt<'_>,
        sp: Span,
        input: TokenStream,
        def_span: Option<Span>,
    ) -> Box<dyn MacResult + 'cx> {
        if !self.valid {
            return DummyResult::any(sp);
        }
        generic_extension(cx, sp, def_span, self.name, input, &self.lhses, &self.rhses)
    }
}

fn trace_macros_note(cx: &mut ExtCtxt<'_>, sp: Span, message: String) {
    let sp = sp.macro_backtrace().last().map(|trace| trace.call_site).unwrap_or(sp);
    cx.expansions.entry(sp).or_default().push(message);
}

/// Given `lhses` and `rhses`, this is the new macro we create
fn generic_extension<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    def_span: Option<Span>,
    name: ast::Ident,
    arg: TokenStream,
    lhses: &[quoted::TokenTree],
    rhses: &[quoted::TokenTree],
) -> Box<dyn MacResult + 'cx> {
    if cx.trace_macros() {
        trace_macros_note(cx, sp, format!("expanding `{}! {{ {} }}`", name, arg));
    }

    // Which arm's failure should we report? (the one furthest along)
    let mut best_failure: Option<(Token, &str)> = None;

    for (i, lhs) in lhses.iter().enumerate() {
        // try each arm's matchers
        let lhs_tt = match *lhs {
            quoted::TokenTree::Delimited(_, ref delim) => &delim.tts[..],
            _ => cx.span_bug(sp, "malformed macro lhs"),
        };

        match TokenTree::parse(cx, lhs_tt, arg.clone()) {
            Success(named_matches) => {
                let rhs = match rhses[i] {
                    // ignore delimiters
                    quoted::TokenTree::Delimited(_, ref delimed) => delimed.tts.clone(),
                    _ => cx.span_bug(sp, "malformed macro rhs"),
                };
                let arm_span = rhses[i].span();

                let rhs_spans = rhs.iter().map(|t| t.span()).collect::<Vec<_>>();
                // rhs has holes ( `$id` and `$(...)` that need filled)
                let mut tts = transcribe(cx, &named_matches, rhs);

                // Replace all the tokens for the corresponding positions in the macro, to maintain
                // proper positions in error reporting, while maintaining the macro_backtrace.
                if rhs_spans.len() == tts.len() {
                    tts = tts.map_enumerated(|i, mut tt| {
                        let mut sp = rhs_spans[i];
                        sp = sp.with_ctxt(tt.span().ctxt());
                        tt.set_span(sp);
                        tt
                    });
                }

                if cx.trace_macros() {
                    trace_macros_note(cx, sp, format!("to `{}`", tts));
                }

                let directory = Directory {
                    path: Cow::from(cx.current_expansion.module.directory.as_path()),
                    ownership: cx.current_expansion.directory_ownership,
                };
                let mut p = Parser::new(cx.parse_sess(), tts, Some(directory), true, false, None);
                p.root_module_name =
                    cx.current_expansion.module.mod_path.last().map(|id| id.as_str().to_string());

                p.process_potential_macro_variable();
                // Let the context choose how to interpret the result.
                // Weird, but useful for X-macros.
                return Box::new(ParserAnyMacro {
                    parser: p,

                    // Pass along the original expansion site and the name of the macro
                    // so we can print a useful error message if the parse of the expanded
                    // macro leaves unparsed tokens.
                    site_span: sp,
                    macro_ident: name,
                    arm_span,
                });
            }
            Failure(token, msg) => match best_failure {
                Some((ref best_token, _)) if best_token.span.lo() >= token.span.lo() => {}
                _ => best_failure = Some((token, msg)),
            },
            Error(err_sp, ref msg) => cx.span_fatal(err_sp.substitute_dummy(sp), &msg[..]),
        }
    }

    let (token, label) = best_failure.expect("ran no matchers");
    let span = token.span.substitute_dummy(sp);
    let mut err = cx.struct_span_err(span, &parse_failure_msg(&token));
    err.span_label(span, label);
    if let Some(sp) = def_span {
        if cx.source_map().span_to_filename(sp).is_real() && !sp.is_dummy() {
            err.span_label(cx.source_map().def_span(sp), "when calling this macro");
        }
    }

    // Check whether there's a missing comma in this macro call, like `println!("{}" a);`
    if let Some((arg, comma_span)) = arg.add_comma() {
        for lhs in lhses {
            // try each arm's matchers
            let lhs_tt = match *lhs {
                quoted::TokenTree::Delimited(_, ref delim) => &delim.tts[..],
                _ => continue,
            };
            match TokenTree::parse(cx, lhs_tt, arg.clone()) {
                Success(_) => {
                    if comma_span.is_dummy() {
                        err.note("you might be missing a comma");
                    } else {
                        err.span_suggestion_short(
                            comma_span,
                            "missing comma here",
                            ", ".to_string(),
                            Applicability::MachineApplicable,
                        );
                    }
                }
                _ => {}
            }
        }
    }
    err.emit();
    cx.trace_macros_diag();
    DummyResult::any(sp)
}

// Note that macro-by-example's input is also matched against a token tree:
//                   $( $lhs:tt => $rhs:tt );+
//
// Holy self-referential!

/// Converts a `macro_rules!` invocation into a syntax extension.
pub fn compile(
    sess: &ParseSess,
    features: &Features,
    def: &ast::Item,
    edition: Edition,
) -> SyntaxExtension {
    let lhs_nm = ast::Ident::new(sym::lhs, def.span);
    let rhs_nm = ast::Ident::new(sym::rhs, def.span);
    let tt_spec = ast::Ident::new(sym::tt, def.span);

    // Parse the macro_rules! invocation
    let body = match def.node {
        ast::ItemKind::MacroDef(ref body) => body,
        _ => unreachable!(),
    };

    // The pattern that macro_rules matches.
    // The grammar for macro_rules! is:
    // $( $lhs:tt => $rhs:tt );+
    // ...quasiquoting this would be nice.
    // These spans won't matter, anyways
    let argument_gram = vec![
        quoted::TokenTree::Sequence(
            DelimSpan::dummy(),
            Lrc::new(quoted::SequenceRepetition {
                tts: vec![
                    quoted::TokenTree::MetaVarDecl(def.span, lhs_nm, tt_spec),
                    quoted::TokenTree::token(token::FatArrow, def.span),
                    quoted::TokenTree::MetaVarDecl(def.span, rhs_nm, tt_spec),
                ],
                separator: Some(Token::new(
                    if body.legacy { token::Semi } else { token::Comma },
                    def.span,
                )),
                op: quoted::KleeneOp::OneOrMore,
                num_captures: 2,
            }),
        ),
        // to phase into semicolon-termination instead of semicolon-separation
        quoted::TokenTree::Sequence(
            DelimSpan::dummy(),
            Lrc::new(quoted::SequenceRepetition {
                tts: vec![quoted::TokenTree::token(token::Semi, def.span)],
                separator: None,
                op: quoted::KleeneOp::ZeroOrMore,
                num_captures: 0,
            }),
        ),
    ];

    let argument_map = match parse(sess, body.stream(), &argument_gram, None, true) {
        Success(m) => m,
        Failure(token, msg) => {
            let s = parse_failure_msg(&token);
            let sp = token.span.substitute_dummy(def.span);
            let mut err = sess.span_diagnostic.struct_span_fatal(sp, &s);
            err.span_label(sp, msg);
            err.emit();
            FatalError.raise();
        }
        Error(sp, s) => {
            sess.span_diagnostic.span_fatal(sp.substitute_dummy(def.span), &s).raise();
        }
    };

    let mut valid = true;

    // Extract the arguments:
    let lhses = match *argument_map[&lhs_nm] {
        MatchedSeq(ref s, _) => s
            .iter()
            .map(|m| {
                if let MatchedNonterminal(ref nt) = *m {
                    if let NtTT(ref tt) = **nt {
                        let tt = quoted::parse(
                            tt.clone().into(),
                            true,
                            sess,
                            features,
                            &def.attrs,
                            edition,
                            def.id,
                        )
                        .pop()
                        .unwrap();
                        valid &= check_lhs_nt_follows(sess, features, &def.attrs, &tt);
                        return tt;
                    }
                }
                sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
            })
            .collect::<Vec<quoted::TokenTree>>(),
        _ => sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs"),
    };

    let rhses = match *argument_map[&rhs_nm] {
        MatchedSeq(ref s, _) => s
            .iter()
            .map(|m| {
                if let MatchedNonterminal(ref nt) = *m {
                    if let NtTT(ref tt) = **nt {
                        return quoted::parse(
                            tt.clone().into(),
                            false,
                            sess,
                            features,
                            &def.attrs,
                            edition,
                            def.id,
                        )
                        .pop()
                        .unwrap();
                    }
                }
                sess.span_diagnostic.span_bug(def.span, "wrong-structured lhs")
            })
            .collect::<Vec<quoted::TokenTree>>(),
        _ => sess.span_diagnostic.span_bug(def.span, "wrong-structured rhs"),
    };

    for rhs in &rhses {
        valid &= check_rhs(sess, rhs);
    }

    // don't abort iteration early, so that errors for multiple lhses can be reported
    for lhs in &lhses {
        valid &= check_lhs_no_empty_seq(sess, slice::from_ref(lhs));
        valid &= check_lhs_duplicate_matcher_bindings(
            sess,
            slice::from_ref(lhs),
            &mut FxHashMap::default(),
            def.id,
        );
    }

    let expander: Box<_> =
        Box::new(MacroRulesMacroExpander { name: def.ident, lhses, rhses, valid });

    let default_transparency = if attr::contains_name(&def.attrs, sym::rustc_transparent_macro) {
        Transparency::Transparent
    } else if body.legacy {
        Transparency::SemiTransparent
    } else {
        Transparency::Opaque
    };

    let allow_internal_unstable =
        attr::find_by_name(&def.attrs, sym::allow_internal_unstable).map(|attr| {
            attr.meta_item_list()
                .map(|list| {
                    list.iter()
                        .filter_map(|it| {
                            let name = it.ident().map(|ident| ident.name);
                            if name.is_none() {
                                sess.span_diagnostic.span_err(
                                    it.span(),
                                    "allow internal unstable expects feature names",
                                )
                            }
                            name
                        })
                        .collect::<Vec<Symbol>>()
                        .into()
                })
                .unwrap_or_else(|| {
                    sess.span_diagnostic.span_warn(
                        attr.span,
                        "allow_internal_unstable expects list of feature names. In the \
                         future this will become a hard error. Please use `allow_internal_unstable(\
                         foo, bar)` to only allow the `foo` and `bar` features",
                    );
                    vec![sym::allow_internal_unstable_backcompat_hack].into()
                })
        });

    let allow_internal_unsafe = attr::contains_name(&def.attrs, sym::allow_internal_unsafe);

    let mut local_inner_macros = false;
    if let Some(macro_export) = attr::find_by_name(&def.attrs, sym::macro_export) {
        if let Some(l) = macro_export.meta_item_list() {
            local_inner_macros = attr::list_contains_name(&l, sym::local_inner_macros);
        }
    }

    let unstable_feature =
        attr::find_stability(&sess, &def.attrs, def.span).and_then(|stability| {
            if let attr::StabilityLevel::Unstable { issue, .. } = stability.level {
                Some((stability.feature, issue))
            } else {
                None
            }
        });

    SyntaxExtension {
        kind: SyntaxExtensionKind::LegacyBang(expander),
        def_info: Some((def.id, def.span)),
        default_transparency,
        allow_internal_unstable,
        allow_internal_unsafe,
        local_inner_macros,
        unstable_feature,
        helper_attrs: Vec::new(),
        edition,
    }
}

fn check_lhs_nt_follows(
    sess: &ParseSess,
    features: &Features,
    attrs: &[ast::Attribute],
    lhs: &quoted::TokenTree,
) -> bool {
    // lhs is going to be like TokenTree::Delimited(...), where the
    // entire lhs is those tts. Or, it can be a "bare sequence", not wrapped in parens.
    if let quoted::TokenTree::Delimited(_, ref tts) = *lhs {
        check_matcher(sess, features, attrs, &tts.tts)
    } else {
        let msg = "invalid macro matcher; matchers must be contained in balanced delimiters";
        sess.span_diagnostic.span_err(lhs.span(), msg);
        false
    }
    // we don't abort on errors on rejection, the driver will do that for us
    // after parsing/expansion. we can report every error in every macro this way.
}

/// Checks that the lhs contains no repetition which could match an empty token
/// tree, because then the matcher would hang indefinitely.
fn check_lhs_no_empty_seq(sess: &ParseSess, tts: &[quoted::TokenTree]) -> bool {
    use quoted::TokenTree;
    for tt in tts {
        match *tt {
            TokenTree::Token(..) | TokenTree::MetaVar(..) | TokenTree::MetaVarDecl(..) => (),
            TokenTree::Delimited(_, ref del) => {
                if !check_lhs_no_empty_seq(sess, &del.tts) {
                    return false;
                }
            }
            TokenTree::Sequence(span, ref seq) => {
                if seq.separator.is_none()
                    && seq.tts.iter().all(|seq_tt| match *seq_tt {
                        TokenTree::MetaVarDecl(_, _, id) => id.name == sym::vis,
                        TokenTree::Sequence(_, ref sub_seq) => {
                            sub_seq.op == quoted::KleeneOp::ZeroOrMore
                                || sub_seq.op == quoted::KleeneOp::ZeroOrOne
                        }
                        _ => false,
                    })
                {
                    let sp = span.entire();
                    sess.span_diagnostic.span_err(sp, "repetition matches empty token tree");
                    return false;
                }
                if !check_lhs_no_empty_seq(sess, &seq.tts) {
                    return false;
                }
            }
        }
    }

    true
}

/// Check that the LHS contains no duplicate matcher bindings. e.g. `$a:expr, $a:expr` would be
/// illegal, since it would be ambiguous which `$a` to use if we ever needed to.
fn check_lhs_duplicate_matcher_bindings(
    sess: &ParseSess,
    tts: &[quoted::TokenTree],
    metavar_names: &mut FxHashMap<Ident, Span>,
    node_id: ast::NodeId,
) -> bool {
    use self::quoted::TokenTree;
    for tt in tts {
        match *tt {
            TokenTree::MetaVarDecl(span, name, _kind) => {
                if let Some(&prev_span) = metavar_names.get(&name) {
                    sess.span_diagnostic
                        .struct_span_err(span, "duplicate matcher binding")
                        .span_note(prev_span, "previous declaration was here")
                        .emit();
                    return false;
                } else {
                    metavar_names.insert(name, span);
                }
            }
            TokenTree::Delimited(_, ref del) => {
                if !check_lhs_duplicate_matcher_bindings(sess, &del.tts, metavar_names, node_id) {
                    return false;
                }
            }
            TokenTree::Sequence(_, ref seq) => {
                if !check_lhs_duplicate_matcher_bindings(sess, &seq.tts, metavar_names, node_id) {
                    return false;
                }
            }
            _ => {}
        }
    }

    true
}

fn check_rhs(sess: &ParseSess, rhs: &quoted::TokenTree) -> bool {
    match *rhs {
        quoted::TokenTree::Delimited(..) => return true,
        _ => sess.span_diagnostic.span_err(rhs.span(), "macro rhs must be delimited"),
    }
    false
}

fn check_matcher(
    sess: &ParseSess,
    features: &Features,
    attrs: &[ast::Attribute],
    matcher: &[quoted::TokenTree],
) -> bool {
    let first_sets = FirstSets::new(matcher);
    let empty_suffix = TokenSet::empty();
    let err = sess.span_diagnostic.err_count();
    check_matcher_core(sess, features, attrs, &first_sets, matcher, &empty_suffix);
    err == sess.span_diagnostic.err_count()
}

// `The FirstSets` for a matcher is a mapping from subsequences in the
// matcher to the FIRST set for that subsequence.
//
// This mapping is partially precomputed via a backwards scan over the
// token trees of the matcher, which provides a mapping from each
// repetition sequence to its *first* set.
//
// (Hypothetically, sequences should be uniquely identifiable via their
// spans, though perhaps that is false, e.g., for macro-generated macros
// that do not try to inject artificial span information. My plan is
// to try to catch such cases ahead of time and not include them in
// the precomputed mapping.)
struct FirstSets {
    // this maps each TokenTree::Sequence `$(tt ...) SEP OP` that is uniquely identified by its
    // span in the original matcher to the First set for the inner sequence `tt ...`.
    //
    // If two sequences have the same span in a matcher, then map that
    // span to None (invalidating the mapping here and forcing the code to
    // use a slow path).
    first: FxHashMap<Span, Option<TokenSet>>,
}

impl FirstSets {
    fn new(tts: &[quoted::TokenTree]) -> FirstSets {
        use quoted::TokenTree;

        let mut sets = FirstSets { first: FxHashMap::default() };
        build_recur(&mut sets, tts);
        return sets;

        // walks backward over `tts`, returning the FIRST for `tts`
        // and updating `sets` at the same time for all sequence
        // substructure we find within `tts`.
        fn build_recur(sets: &mut FirstSets, tts: &[TokenTree]) -> TokenSet {
            let mut first = TokenSet::empty();
            for tt in tts.iter().rev() {
                match *tt {
                    TokenTree::Token(..) | TokenTree::MetaVar(..) | TokenTree::MetaVarDecl(..) => {
                        first.replace_with(tt.clone());
                    }
                    TokenTree::Delimited(span, ref delimited) => {
                        build_recur(sets, &delimited.tts[..]);
                        first.replace_with(delimited.open_tt(span.open));
                    }
                    TokenTree::Sequence(sp, ref seq_rep) => {
                        let subfirst = build_recur(sets, &seq_rep.tts[..]);

                        match sets.first.entry(sp.entire()) {
                            Entry::Vacant(vac) => {
                                vac.insert(Some(subfirst.clone()));
                            }
                            Entry::Occupied(mut occ) => {
                                // if there is already an entry, then a span must have collided.
                                // This should not happen with typical macro_rules macros,
                                // but syntax extensions need not maintain distinct spans,
                                // so distinct syntax trees can be assigned the same span.
                                // In such a case, the map cannot be trusted; so mark this
                                // entry as unusable.
                                occ.insert(None);
                            }
                        }

                        // If the sequence contents can be empty, then the first
                        // token could be the separator token itself.

                        if let (Some(sep), true) = (&seq_rep.separator, subfirst.maybe_empty) {
                            first.add_one_maybe(TokenTree::Token(sep.clone()));
                        }

                        // Reverse scan: Sequence comes before `first`.
                        if subfirst.maybe_empty
                            || seq_rep.op == quoted::KleeneOp::ZeroOrMore
                            || seq_rep.op == quoted::KleeneOp::ZeroOrOne
                        {
                            // If sequence is potentially empty, then
                            // union them (preserving first emptiness).
                            first.add_all(&TokenSet { maybe_empty: true, ..subfirst });
                        } else {
                            // Otherwise, sequence guaranteed
                            // non-empty; replace first.
                            first = subfirst;
                        }
                    }
                }
            }

            first
        }
    }

    // walks forward over `tts` until all potential FIRST tokens are
    // identified.
    fn first(&self, tts: &[quoted::TokenTree]) -> TokenSet {
        use quoted::TokenTree;

        let mut first = TokenSet::empty();
        for tt in tts.iter() {
            assert!(first.maybe_empty);
            match *tt {
                TokenTree::Token(..) | TokenTree::MetaVar(..) | TokenTree::MetaVarDecl(..) => {
                    first.add_one(tt.clone());
                    return first;
                }
                TokenTree::Delimited(span, ref delimited) => {
                    first.add_one(delimited.open_tt(span.open));
                    return first;
                }
                TokenTree::Sequence(sp, ref seq_rep) => {
                    match self.first.get(&sp.entire()) {
                        Some(&Some(ref subfirst)) => {
                            // If the sequence contents can be empty, then the first
                            // token could be the separator token itself.

                            if let (Some(sep), true) = (&seq_rep.separator, subfirst.maybe_empty) {
                                first.add_one_maybe(TokenTree::Token(sep.clone()));
                            }

                            assert!(first.maybe_empty);
                            first.add_all(subfirst);
                            if subfirst.maybe_empty
                                || seq_rep.op == quoted::KleeneOp::ZeroOrMore
                                || seq_rep.op == quoted::KleeneOp::ZeroOrOne
                            {
                                // continue scanning for more first
                                // tokens, but also make sure we
                                // restore empty-tracking state
                                first.maybe_empty = true;
                                continue;
                            } else {
                                return first;
                            }
                        }

                        Some(&None) => {
                            panic!("assume all sequences have (unique) spans for now");
                        }

                        None => {
                            panic!("We missed a sequence during FirstSets construction");
                        }
                    }
                }
            }
        }

        // we only exit the loop if `tts` was empty or if every
        // element of `tts` matches the empty sequence.
        assert!(first.maybe_empty);
        first
    }
}

// A set of `quoted::TokenTree`s, which may include `TokenTree::Match`s
// (for macro-by-example syntactic variables). It also carries the
// `maybe_empty` flag; that is true if and only if the matcher can
// match an empty token sequence.
//
// The First set is computed on submatchers like `$($a:expr b),* $(c)* d`,
// which has corresponding FIRST = {$a:expr, c, d}.
// Likewise, `$($a:expr b),* $(c)+ d` has FIRST = {$a:expr, c}.
//
// (Notably, we must allow for *-op to occur zero times.)
#[derive(Clone, Debug)]
struct TokenSet {
    tokens: Vec<quoted::TokenTree>,
    maybe_empty: bool,
}

impl TokenSet {
    // Returns a set for the empty sequence.
    fn empty() -> Self {
        TokenSet { tokens: Vec::new(), maybe_empty: true }
    }

    // Returns the set `{ tok }` for the single-token (and thus
    // non-empty) sequence [tok].
    fn singleton(tok: quoted::TokenTree) -> Self {
        TokenSet { tokens: vec![tok], maybe_empty: false }
    }

    // Changes self to be the set `{ tok }`.
    // Since `tok` is always present, marks self as non-empty.
    fn replace_with(&mut self, tok: quoted::TokenTree) {
        self.tokens.clear();
        self.tokens.push(tok);
        self.maybe_empty = false;
    }

    // Changes self to be the empty set `{}`; meant for use when
    // the particular token does not matter, but we want to
    // record that it occurs.
    fn replace_with_irrelevant(&mut self) {
        self.tokens.clear();
        self.maybe_empty = false;
    }

    // Adds `tok` to the set for `self`, marking sequence as non-empy.
    fn add_one(&mut self, tok: quoted::TokenTree) {
        if !self.tokens.contains(&tok) {
            self.tokens.push(tok);
        }
        self.maybe_empty = false;
    }

    // Adds `tok` to the set for `self`. (Leaves `maybe_empty` flag alone.)
    fn add_one_maybe(&mut self, tok: quoted::TokenTree) {
        if !self.tokens.contains(&tok) {
            self.tokens.push(tok);
        }
    }

    // Adds all elements of `other` to this.
    //
    // (Since this is a set, we filter out duplicates.)
    //
    // If `other` is potentially empty, then preserves the previous
    // setting of the empty flag of `self`. If `other` is guaranteed
    // non-empty, then `self` is marked non-empty.
    fn add_all(&mut self, other: &Self) {
        for tok in &other.tokens {
            if !self.tokens.contains(tok) {
                self.tokens.push(tok.clone());
            }
        }
        if !other.maybe_empty {
            self.maybe_empty = false;
        }
    }
}

// Checks that `matcher` is internally consistent and that it
// can legally by followed by a token N, for all N in `follow`.
// (If `follow` is empty, then it imposes no constraint on
// the `matcher`.)
//
// Returns the set of NT tokens that could possibly come last in
// `matcher`. (If `matcher` matches the empty sequence, then
// `maybe_empty` will be set to true.)
//
// Requires that `first_sets` is pre-computed for `matcher`;
// see `FirstSets::new`.
fn check_matcher_core(
    sess: &ParseSess,
    features: &Features,
    attrs: &[ast::Attribute],
    first_sets: &FirstSets,
    matcher: &[quoted::TokenTree],
    follow: &TokenSet,
) -> TokenSet {
    use quoted::TokenTree;

    let mut last = TokenSet::empty();

    // 2. For each token and suffix  [T, SUFFIX] in M:
    // ensure that T can be followed by SUFFIX, and if SUFFIX may be empty,
    // then ensure T can also be followed by any element of FOLLOW.
    'each_token: for i in 0..matcher.len() {
        let token = &matcher[i];
        let suffix = &matcher[i + 1..];

        let build_suffix_first = || {
            let mut s = first_sets.first(suffix);
            if s.maybe_empty {
                s.add_all(follow);
            }
            s
        };

        // (we build `suffix_first` on demand below; you can tell
        // which cases are supposed to fall through by looking for the
        // initialization of this variable.)
        let suffix_first;

        // First, update `last` so that it corresponds to the set
        // of NT tokens that might end the sequence `... token`.
        match *token {
            TokenTree::Token(..) | TokenTree::MetaVar(..) | TokenTree::MetaVarDecl(..) => {
                let can_be_followed_by_any;
                if let Err(bad_frag) = has_legal_fragment_specifier(sess, features, attrs, token) {
                    let msg = format!("invalid fragment specifier `{}`", bad_frag);
                    sess.span_diagnostic
                        .struct_span_err(token.span(), &msg)
                        .help(VALID_FRAGMENT_NAMES_MSG)
                        .emit();
                    // (This eliminates false positives and duplicates
                    // from error messages.)
                    can_be_followed_by_any = true;
                } else {
                    can_be_followed_by_any = token_can_be_followed_by_any(token);
                }

                if can_be_followed_by_any {
                    // don't need to track tokens that work with any,
                    last.replace_with_irrelevant();
                    // ... and don't need to check tokens that can be
                    // followed by anything against SUFFIX.
                    continue 'each_token;
                } else {
                    last.replace_with(token.clone());
                    suffix_first = build_suffix_first();
                }
            }
            TokenTree::Delimited(span, ref d) => {
                let my_suffix = TokenSet::singleton(d.close_tt(span.close));
                check_matcher_core(sess, features, attrs, first_sets, &d.tts, &my_suffix);
                // don't track non NT tokens
                last.replace_with_irrelevant();

                // also, we don't need to check delimited sequences
                // against SUFFIX
                continue 'each_token;
            }
            TokenTree::Sequence(_, ref seq_rep) => {
                suffix_first = build_suffix_first();
                // The trick here: when we check the interior, we want
                // to include the separator (if any) as a potential
                // (but not guaranteed) element of FOLLOW. So in that
                // case, we make a temp copy of suffix and stuff
                // delimiter in there.
                //
                // FIXME: Should I first scan suffix_first to see if
                // delimiter is already in it before I go through the
                // work of cloning it? But then again, this way I may
                // get a "tighter" span?
                let mut new;
                let my_suffix = if let Some(sep) = &seq_rep.separator {
                    new = suffix_first.clone();
                    new.add_one_maybe(TokenTree::Token(sep.clone()));
                    &new
                } else {
                    &suffix_first
                };

                // At this point, `suffix_first` is built, and
                // `my_suffix` is some TokenSet that we can use
                // for checking the interior of `seq_rep`.
                let next =
                    check_matcher_core(sess, features, attrs, first_sets, &seq_rep.tts, my_suffix);
                if next.maybe_empty {
                    last.add_all(&next);
                } else {
                    last = next;
                }

                // the recursive call to check_matcher_core already ran the 'each_last
                // check below, so we can just keep going forward here.
                continue 'each_token;
            }
        }

        // (`suffix_first` guaranteed initialized once reaching here.)

        // Now `last` holds the complete set of NT tokens that could
        // end the sequence before SUFFIX. Check that every one works with `suffix`.
        'each_last: for token in &last.tokens {
            if let TokenTree::MetaVarDecl(_, ref name, ref frag_spec) = *token {
                for next_token in &suffix_first.tokens {
                    match is_in_follow(next_token, &frag_spec.as_str()) {
                        IsInFollow::Invalid(msg, help) => {
                            sess.span_diagnostic
                                .struct_span_err(next_token.span(), &msg)
                                .help(help)
                                .emit();
                            // don't bother reporting every source of
                            // conflict for a particular element of `last`.
                            continue 'each_last;
                        }
                        IsInFollow::Yes => {}
                        IsInFollow::No(possible) => {
                            let may_be = if last.tokens.len() == 1 && suffix_first.tokens.len() == 1
                            {
                                "is"
                            } else {
                                "may be"
                            };

                            let sp = next_token.span();
                            let mut err = sess.span_diagnostic.struct_span_err(
                                sp,
                                &format!(
                                    "`${name}:{frag}` {may_be} followed by `{next}`, which \
                                     is not allowed for `{frag}` fragments",
                                    name = name,
                                    frag = frag_spec,
                                    next = quoted_tt_to_string(next_token),
                                    may_be = may_be
                                ),
                            );
                            err.span_label(
                                sp,
                                format!("not allowed after `{}` fragments", frag_spec),
                            );
                            let msg = "allowed there are: ";
                            match possible {
                                &[] => {}
                                &[t] => {
                                    err.note(&format!(
                                        "only {} is allowed after `{}` fragments",
                                        t, frag_spec,
                                    ));
                                }
                                ts => {
                                    err.note(&format!(
                                        "{}{} or {}",
                                        msg,
                                        ts[..ts.len() - 1]
                                            .iter()
                                            .map(|s| *s)
                                            .collect::<Vec<_>>()
                                            .join(", "),
                                        ts[ts.len() - 1],
                                    ));
                                }
                            }
                            err.emit();
                        }
                    }
                }
            }
        }
    }
    last
}

fn token_can_be_followed_by_any(tok: &quoted::TokenTree) -> bool {
    if let quoted::TokenTree::MetaVarDecl(_, _, frag_spec) = *tok {
        frag_can_be_followed_by_any(&frag_spec.as_str())
    } else {
        // (Non NT's can always be followed by anthing in matchers.)
        true
    }
}

/// Returns `true` if a fragment of type `frag` can be followed by any sort of
/// token. We use this (among other things) as a useful approximation
/// for when `frag` can be followed by a repetition like `$(...)*` or
/// `$(...)+`. In general, these can be a bit tricky to reason about,
/// so we adopt a conservative position that says that any fragment
/// specifier which consumes at most one token tree can be followed by
/// a fragment specifier (indeed, these fragments can be followed by
/// ANYTHING without fear of future compatibility hazards).
fn frag_can_be_followed_by_any(frag: &str) -> bool {
    match frag {
        "item"     | // always terminated by `}` or `;`
        "block"    | // exactly one token tree
        "ident"    | // exactly one token tree
        "literal"  | // exactly one token tree
        "meta"     | // exactly one token tree
        "lifetime" | // exactly one token tree
        "tt" =>   // exactly one token tree
            true,

        _ =>
            false,
    }
}

enum IsInFollow {
    Yes,
    No(&'static [&'static str]),
    Invalid(String, &'static str),
}

/// Returns `true` if `frag` can legally be followed by the token `tok`. For
/// fragments that can consume an unbounded number of tokens, `tok`
/// must be within a well-defined follow set. This is intended to
/// guarantee future compatibility: for example, without this rule, if
/// we expanded `expr` to include a new binary operator, we might
/// break macros that were relying on that binary operator as a
/// separator.
// when changing this do not forget to update doc/book/macros.md!
fn is_in_follow(tok: &quoted::TokenTree, frag: &str) -> IsInFollow {
    use quoted::TokenTree;

    if let TokenTree::Token(Token { kind: token::CloseDelim(_), .. }) = *tok {
        // closing a token tree can never be matched by any fragment;
        // iow, we always require that `(` and `)` match, etc.
        IsInFollow::Yes
    } else {
        match frag {
            "item" => {
                // since items *must* be followed by either a `;` or a `}`, we can
                // accept anything after them
                IsInFollow::Yes
            }
            "block" => {
                // anything can follow block, the braces provide an easy boundary to
                // maintain
                IsInFollow::Yes
            }
            "stmt" | "expr" => {
                const TOKENS: &[&str] = &["`=>`", "`,`", "`;`"];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        FatArrow | Comma | Semi => IsInFollow::Yes,
                        _ => IsInFollow::No(TOKENS),
                    },
                    _ => IsInFollow::No(TOKENS),
                }
            }
            "pat" => {
                const TOKENS: &[&str] = &["`=>`", "`,`", "`=`", "`|`", "`if`", "`in`"];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        FatArrow | Comma | Eq | BinOp(token::Or) => IsInFollow::Yes,
                        Ident(name, false) if name == kw::If || name == kw::In => IsInFollow::Yes,
                        _ => IsInFollow::No(TOKENS),
                    },
                    _ => IsInFollow::No(TOKENS),
                }
            }
            "path" | "ty" => {
                const TOKENS: &[&str] = &[
                    "`{`", "`[`", "`=>`", "`,`", "`>`", "`=`", "`:`", "`;`", "`|`", "`as`",
                    "`where`",
                ];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        OpenDelim(token::DelimToken::Brace)
                        | OpenDelim(token::DelimToken::Bracket)
                        | Comma
                        | FatArrow
                        | Colon
                        | Eq
                        | Gt
                        | BinOp(token::Shr)
                        | Semi
                        | BinOp(token::Or) => IsInFollow::Yes,
                        Ident(name, false) if name == kw::As || name == kw::Where => {
                            IsInFollow::Yes
                        }
                        _ => IsInFollow::No(TOKENS),
                    },
                    TokenTree::MetaVarDecl(_, _, frag) if frag.name == sym::block => {
                        IsInFollow::Yes
                    }
                    _ => IsInFollow::No(TOKENS),
                }
            }
            "ident" | "lifetime" => {
                // being a single token, idents and lifetimes are harmless
                IsInFollow::Yes
            }
            "literal" => {
                // literals may be of a single token, or two tokens (negative numbers)
                IsInFollow::Yes
            }
            "meta" | "tt" => {
                // being either a single token or a delimited sequence, tt is
                // harmless
                IsInFollow::Yes
            }
            "vis" => {
                // Explicitly disallow `priv`, on the off chance it comes back.
                const TOKENS: &[&str] = &["`,`", "an ident", "a type"];
                match tok {
                    TokenTree::Token(token) => match token.kind {
                        Comma => IsInFollow::Yes,
                        Ident(name, is_raw) if is_raw || name != kw::Priv => IsInFollow::Yes,
                        _ => {
                            if token.can_begin_type() {
                                IsInFollow::Yes
                            } else {
                                IsInFollow::No(TOKENS)
                            }
                        }
                    },
                    TokenTree::MetaVarDecl(_, _, frag)
                        if frag.name == sym::ident
                            || frag.name == sym::ty
                            || frag.name == sym::path =>
                    {
                        IsInFollow::Yes
                    }
                    _ => IsInFollow::No(TOKENS),
                }
            }
            "" => IsInFollow::Yes, // kw::Invalid
            _ => IsInFollow::Invalid(
                format!("invalid fragment specifier `{}`", frag),
                VALID_FRAGMENT_NAMES_MSG,
            ),
        }
    }
}

fn has_legal_fragment_specifier(
    sess: &ParseSess,
    features: &Features,
    attrs: &[ast::Attribute],
    tok: &quoted::TokenTree,
) -> Result<(), String> {
    debug!("has_legal_fragment_specifier({:?})", tok);
    if let quoted::TokenTree::MetaVarDecl(_, _, ref frag_spec) = *tok {
        let frag_span = tok.span();
        if !is_legal_fragment_specifier(sess, features, attrs, frag_spec.name, frag_span) {
            return Err(frag_spec.to_string());
        }
    }
    Ok(())
}

fn is_legal_fragment_specifier(
    _sess: &ParseSess,
    _features: &Features,
    _attrs: &[ast::Attribute],
    frag_name: Symbol,
    _frag_span: Span,
) -> bool {
    /*
     * If new fragment specifiers are invented in nightly, `_sess`,
     * `_features`, `_attrs`, and `_frag_span` will be useful here
     * for checking against feature gates. See past versions of
     * this function.
     */
    match frag_name {
        sym::item
        | sym::block
        | sym::stmt
        | sym::expr
        | sym::pat
        | sym::lifetime
        | sym::path
        | sym::ty
        | sym::ident
        | sym::meta
        | sym::tt
        | sym::vis
        | sym::literal
        | kw::Invalid => true,
        _ => false,
    }
}

fn quoted_tt_to_string(tt: &quoted::TokenTree) -> String {
    match *tt {
        quoted::TokenTree::Token(ref token) => crate::print::pprust::token_to_string(&token),
        quoted::TokenTree::MetaVar(_, name) => format!("${}", name),
        quoted::TokenTree::MetaVarDecl(_, name, kind) => format!("${}:{}", name, kind),
        _ => panic!(
            "unexpected quoted::TokenTree::{{Sequence or Delimited}} \
             in follow set checker"
        ),
    }
}
