//! This module takes a (parsed) definition of `macro_rules` invocation, a
//! `tt::TokenTree` representing an argument of macro invocation, and produces a
//! `tt::TokenTree` for the result of the expansion.

use ra_parser::FragmentKind::*;
use ra_syntax::SmolStr;
use rustc_hash::FxHashMap;
use tt::TokenId;

use crate::tt_cursor::TtCursor;
use crate::ExpandError;

pub(crate) fn expand(
    rules: &crate::MacroRules,
    input: &tt::Subtree,
) -> Result<tt::Subtree, ExpandError> {
    rules.rules.iter().find_map(|it| expand_rule(it, input).ok()).ok_or(ExpandError::NoMatchingRule)
}

fn expand_rule(rule: &crate::Rule, input: &tt::Subtree) -> Result<tt::Subtree, ExpandError> {
    let mut input = TtCursor::new(input);
    let bindings = match_lhs(&rule.lhs, &mut input)?;
    if !input.is_eof() {
        return Err(ExpandError::UnexpectedToken);
    }

    let mut ctx = ExpandCtx { bindings: &bindings, nesting: Vec::new(), var_expanded: false };

    expand_subtree(&rule.rhs, &mut ctx)
}

/// The actual algorithm for expansion is not too hard, but is pretty tricky.
/// `Bindings` structure is the key to understanding what we are doing here.
///
/// On the high level, it stores mapping from meta variables to the bits of
/// syntax it should be substituted with. For example, if `$e:expr` is matched
/// with `1 + 1` by macro_rules, the `Binding` will store `$e -> 1 + 1`.
///
/// The tricky bit is dealing with repetitions (`$()*`). Consider this example:
///
/// ```not_rust
/// macro_rules! foo {
///     ($($ i:ident $($ e:expr),*);*) => {
///         $(fn $ i() { $($ e);*; })*
///     }
/// }
/// foo! { foo 1,2,3; bar 4,5,6 }
/// ```
///
/// Here, the `$i` meta variable is matched first with `foo` and then with
/// `bar`, and `$e` is matched in turn with `1`, `2`, `3`, `4`, `5`, `6`.
///
/// To represent such "multi-mappings", we use a recursive structures: we map
/// variables not to values, but to *lists* of values or other lists (that is,
/// to the trees).
///
/// For the above example, the bindings would store
///
/// ```not_rust
/// i -> [foo, bar]
/// e -> [[1, 2, 3], [4, 5, 6]]
/// ```
///
/// We construct `Bindings` in the `match_lhs`. The interesting case is
/// `TokenTree::Repeat`, where we use `push_nested` to create the desired
/// nesting structure.
///
/// The other side of the puzzle is `expand_subtree`, where we use the bindings
/// to substitute meta variables in the output template. When expanding, we
/// maintain a `nesting` stack of indices which tells us which occurrence from
/// the `Bindings` we should take. We push to the stack when we enter a
/// repetition.
///
/// In other words, `Bindings` is a *multi* mapping from `SmolStr` to
/// `tt::TokenTree`, where the index to select a particular `TokenTree` among
/// many is not a plain `usize`, but an `&[usize]`.
#[derive(Debug, Default)]
struct Bindings {
    inner: FxHashMap<SmolStr, Binding>,
}

#[derive(Debug)]
enum Binding {
    Fragment(Fragment),
    Nested(Vec<Binding>),
    Empty,
}

#[derive(Debug, Clone)]
enum Fragment {
    /// token fragments are just copy-pasted into the output
    Tokens(tt::TokenTree),
    /// Ast fragments are inserted with fake delimiters, so as to make things
    /// like `$i * 2` where `$i = 1 + 1` work as expectd.
    Ast(tt::TokenTree),
}

impl Bindings {
    fn push_optional(&mut self, name: &SmolStr) {
        // FIXME: Do we have a better way to represent an empty token ?
        // Insert an empty subtree for empty token
        let tt = tt::Subtree { delimiter: tt::Delimiter::None, token_trees: vec![] }.into();
        self.inner.insert(name.clone(), Binding::Fragment(Fragment::Tokens(tt)));
    }

    fn push_empty(&mut self, name: &SmolStr) {
        self.inner.insert(name.clone(), Binding::Empty);
    }

    fn contains(&self, name: &SmolStr) -> bool {
        self.inner.contains_key(name)
    }

    fn get(&self, name: &SmolStr, nesting: &[usize]) -> Result<&Fragment, ExpandError> {
        let mut b = self.inner.get(name).ok_or_else(|| {
            ExpandError::BindingError(format!("could not find binding `{}`", name))
        })?;
        for &idx in nesting.iter() {
            b = match b {
                Binding::Fragment(_) => break,
                Binding::Nested(bs) => bs.get(idx).ok_or_else(|| {
                    ExpandError::BindingError(format!("could not find nested binding `{}`", name))
                })?,
                Binding::Empty => {
                    return Err(ExpandError::BindingError(format!(
                        "could not find empty binding `{}`",
                        name
                    )))
                }
            };
        }
        match b {
            Binding::Fragment(it) => Ok(it),
            Binding::Nested(_) => Err(ExpandError::BindingError(format!(
                "expected simple binding, found nested binding `{}`",
                name
            ))),
            Binding::Empty => Err(ExpandError::BindingError(format!(
                "expected simple binding, found empty binding `{}`",
                name
            ))),
        }
    }

    fn push_nested(&mut self, idx: usize, nested: Bindings) -> Result<(), ExpandError> {
        for (key, value) in nested.inner {
            if !self.inner.contains_key(&key) {
                self.inner.insert(key.clone(), Binding::Nested(Vec::new()));
            }
            match self.inner.get_mut(&key) {
                Some(Binding::Nested(it)) => {
                    // insert empty nested bindings before this one
                    while it.len() < idx {
                        it.push(Binding::Nested(vec![]));
                    }
                    it.push(value);
                }
                _ => {
                    return Err(ExpandError::BindingError(format!(
                        "could not find binding `{}`",
                        key
                    )));
                }
            }
        }
        Ok(())
    }

    fn merge(&mut self, nested: Bindings) {
        self.inner.extend(nested.inner);
    }
}

fn collect_vars(subtree: &crate::Subtree) -> Vec<SmolStr> {
    let mut res = vec![];

    for tkn in subtree.token_trees.iter() {
        match tkn {
            crate::TokenTree::Leaf(crate::Leaf::Var(crate::Var { text, .. })) => {
                res.push(text.clone());
            }
            crate::TokenTree::Subtree(subtree) => {
                res.extend(collect_vars(subtree));
            }
            crate::TokenTree::Repeat(crate::Repeat { subtree, .. }) => {
                res.extend(collect_vars(subtree));
            }
            _ => {}
        }
    }

    res
}

fn match_lhs(pattern: &crate::Subtree, input: &mut TtCursor) -> Result<Bindings, ExpandError> {
    let mut res = Bindings::default();
    for pat in pattern.token_trees.iter() {
        match pat {
            crate::TokenTree::Leaf(leaf) => match leaf {
                crate::Leaf::Var(crate::Var { text, kind }) => {
                    let kind = kind.as_ref().ok_or(ExpandError::UnexpectedToken)?;
                    match match_meta_var(kind.as_str(), input)? {
                        Some(fragment) => {
                            res.inner.insert(text.clone(), Binding::Fragment(fragment));
                        }
                        None => res.push_optional(text),
                    }
                }
                crate::Leaf::Punct(punct) => {
                    if !input.eat_punct().map(|p| p.char == punct.char).unwrap_or(false) {
                        return Err(ExpandError::UnexpectedToken);
                    }
                }
                crate::Leaf::Ident(ident) => {
                    if input.eat_ident().map(|i| &i.text) != Some(&ident.text) {
                        return Err(ExpandError::UnexpectedToken);
                    }
                }
                crate::Leaf::Literal(literal) => {
                    if input.eat_literal().map(|i| &i.text) != Some(&literal.text) {
                        return Err(ExpandError::UnexpectedToken);
                    }
                }
            },
            crate::TokenTree::Repeat(crate::Repeat { subtree, kind, separator }) => {
                // Dirty hack to make macro-expansion terminate.
                // This should be replaced by a propper macro-by-example implementation
                let mut limit = 65536;
                let mut counter = 0;

                let mut memento = input.save();

                loop {
                    match match_lhs(subtree, input) {
                        Ok(nested) => {
                            limit -= 1;
                            if limit == 0 {
                                log::warn!("match_lhs excced in repeat pattern exceed limit => {:#?}\n{:#?}\n{:#?}\n{:#?}", subtree, input, kind, separator);
                                break;
                            }

                            memento = input.save();
                            res.push_nested(counter, nested)?;
                            counter += 1;
                            if counter == 1 {
                                if let crate::RepeatKind::ZeroOrOne = kind {
                                    break;
                                }
                            }

                            if let Some(separator) = separator {
                                if !input
                                    .eat_seperator()
                                    .map(|sep| sep == *separator)
                                    .unwrap_or(false)
                                {
                                    input.rollback(memento);
                                    break;
                                }
                            }
                        }
                        Err(_) => {
                            input.rollback(memento);
                            break;
                        }
                    }
                }

                match kind {
                    crate::RepeatKind::OneOrMore if counter == 0 => {
                        return Err(ExpandError::UnexpectedToken);
                    }
                    _ if counter == 0 => {
                        // Collect all empty variables in subtrees
                        collect_vars(subtree).iter().for_each(|s| res.push_empty(s));
                    }
                    _ => {}
                }
            }
            crate::TokenTree::Subtree(subtree) => {
                let input_subtree =
                    input.eat_subtree().map_err(|_| ExpandError::UnexpectedToken)?;
                if subtree.delimiter != input_subtree.delimiter {
                    return Err(ExpandError::UnexpectedToken);
                }

                let mut input = TtCursor::new(input_subtree);
                let bindings = match_lhs(&subtree, &mut input)?;
                if !input.is_eof() {
                    return Err(ExpandError::UnexpectedToken);
                }

                res.merge(bindings);
            }
        }
    }
    Ok(res)
}

fn match_meta_var(kind: &str, input: &mut TtCursor) -> Result<Option<Fragment>, ExpandError> {
    let fragment = match kind {
        "path" => Path,
        "expr" => Expr,
        "ty" => Type,
        "pat" => Pattern,
        "stmt" => Statement,
        "block" => Block,
        "meta" => MetaItem,
        "item" => Item,
        _ => {
            let tt = match kind {
                "ident" => {
                    let ident = input.eat_ident().ok_or(ExpandError::UnexpectedToken)?.clone();
                    tt::Leaf::from(ident).into()
                }
                "tt" => input.eat().ok_or(ExpandError::UnexpectedToken)?.clone(),
                "lifetime" => input.eat_lifetime().ok_or(ExpandError::UnexpectedToken)?.clone(),
                "literal" => {
                    let literal = input.eat_literal().ok_or(ExpandError::UnexpectedToken)?.clone();
                    tt::Leaf::from(literal).into()
                }
                // `vis` is optional
                "vis" => match input.try_eat_vis() {
                    Some(vis) => vis,
                    None => return Ok(None),
                },
                _ => return Err(ExpandError::UnexpectedToken),
            };
            return Ok(Some(Fragment::Tokens(tt)));
        }
    };
    let tt = input.eat_fragment(fragment).ok_or(ExpandError::UnexpectedToken)?;
    let fragment = if kind == "expr" { Fragment::Ast(tt) } else { Fragment::Tokens(tt) };
    Ok(Some(fragment))
}

#[derive(Debug)]
struct ExpandCtx<'a> {
    bindings: &'a Bindings,
    nesting: Vec<usize>,
    var_expanded: bool,
}

fn expand_subtree(
    template: &crate::Subtree,
    ctx: &mut ExpandCtx,
) -> Result<tt::Subtree, ExpandError> {
    let mut buf: Vec<tt::TokenTree> = Vec::new();
    for tt in template.token_trees.iter() {
        let tt = expand_tt(tt, ctx)?;
        push_fragment(&mut buf, tt);
    }

    Ok(tt::Subtree { delimiter: template.delimiter, token_trees: buf })
}

/// Reduce single token subtree to single token
/// In `tt` matcher case, all tt tokens will be braced by a Delimiter::None
/// which makes all sort of problems.
fn reduce_single_token(mut subtree: tt::Subtree) -> tt::TokenTree {
    if subtree.delimiter != tt::Delimiter::None || subtree.token_trees.len() != 1 {
        return subtree.into();
    }

    match subtree.token_trees.pop().unwrap() {
        tt::TokenTree::Subtree(subtree) => reduce_single_token(subtree),
        tt::TokenTree::Leaf(token) => token.into(),
    }
}

fn expand_tt(template: &crate::TokenTree, ctx: &mut ExpandCtx) -> Result<Fragment, ExpandError> {
    let res: tt::TokenTree = match template {
        crate::TokenTree::Subtree(subtree) => expand_subtree(subtree, ctx)?.into(),
        crate::TokenTree::Repeat(repeat) => {
            let mut buf: Vec<tt::TokenTree> = Vec::new();
            ctx.nesting.push(0);
            // Dirty hack to make macro-expansion terminate.
            // This should be replaced by a propper macro-by-example implementation
            let mut limit = 65536;
            let mut has_seps = 0;
            let mut counter = 0;

            // We store the old var expanded value, and restore it later
            // It is because before this `$repeat`,
            // it is possible some variables already expanad in the same subtree
            //
            // `some_var_expanded` keep check if the deeper subtree has expanded variables
            let mut some_var_expanded = false;
            let old_var_expanded = ctx.var_expanded;
            ctx.var_expanded = false;

            while let Ok(t) = expand_subtree(&repeat.subtree, ctx) {
                // if no var expanded in the child, we count it as a fail
                if !ctx.var_expanded {
                    break;
                }

                // Reset `ctx.var_expandeded` to see if there is other expanded variable
                // in the next matching
                some_var_expanded = true;
                ctx.var_expanded = false;

                counter += 1;
                limit -= 1;
                if limit == 0 {
                    log::warn!(
                        "expand_tt excced in repeat pattern exceed limit => {:#?}\n{:#?}",
                        template,
                        ctx
                    );
                    break;
                }

                let idx = ctx.nesting.pop().unwrap();
                ctx.nesting.push(idx + 1);
                push_subtree(&mut buf, t);

                if let Some(ref sep) = repeat.separator {
                    match sep {
                        crate::Separator::Ident(ident) => {
                            has_seps = 1;
                            buf.push(tt::Leaf::from(ident.clone()).into());
                        }
                        crate::Separator::Literal(lit) => {
                            has_seps = 1;
                            buf.push(tt::Leaf::from(lit.clone()).into());
                        }

                        crate::Separator::Puncts(puncts) => {
                            has_seps = puncts.len();
                            for punct in puncts {
                                buf.push(tt::Leaf::from(*punct).into());
                            }
                        }
                    }
                }

                if let crate::RepeatKind::ZeroOrOne = repeat.kind {
                    break;
                }
            }

            // Restore the `var_expanded` by combining old one and the new one
            ctx.var_expanded = some_var_expanded || old_var_expanded;

            ctx.nesting.pop().unwrap();
            for _ in 0..has_seps {
                buf.pop();
            }

            if crate::RepeatKind::OneOrMore == repeat.kind && counter == 0 {
                return Err(ExpandError::UnexpectedToken);
            }

            // Check if it is a single token subtree without any delimiter
            // e.g {Delimiter:None> ['>'] /Delimiter:None>}
            reduce_single_token(tt::Subtree { delimiter: tt::Delimiter::None, token_trees: buf })
        }
        crate::TokenTree::Leaf(leaf) => match leaf {
            crate::Leaf::Ident(ident) => {
                tt::Leaf::from(tt::Ident { text: ident.text.clone(), id: TokenId::unspecified() })
                    .into()
            }
            crate::Leaf::Punct(punct) => tt::Leaf::from(*punct).into(),
            crate::Leaf::Var(v) => {
                if v.text == "crate" {
                    // FIXME: Properly handle $crate token
                    tt::Leaf::from(tt::Ident { text: "$crate".into(), id: TokenId::unspecified() })
                        .into()
                } else if !ctx.bindings.contains(&v.text) {
                    // Note that it is possible to have a `$var` inside a macro which is not bound.
                    // For example:
                    // ```
                    // macro_rules! foo {
                    //     ($a:ident, $b:ident, $c:tt) => {
                    //         macro_rules! bar {
                    //             ($bi:ident) => {
                    //                 fn $bi() -> u8 {$c}
                    //             }
                    //         }
                    //     }
                    // ```
                    // We just treat it a normal tokens
                    tt::Subtree {
                        delimiter: tt::Delimiter::None,
                        token_trees: vec![
                            tt::Leaf::from(tt::Punct { char: '$', spacing: tt::Spacing::Alone })
                                .into(),
                            tt::Leaf::from(tt::Ident {
                                text: v.text.clone(),
                                id: TokenId::unspecified(),
                            })
                            .into(),
                        ],
                    }
                    .into()
                } else {
                    let fragment = ctx.bindings.get(&v.text, &ctx.nesting)?.clone();
                    ctx.var_expanded = true;
                    match fragment {
                        Fragment::Tokens(tt) => {
                            if let tt::TokenTree::Subtree(subtree) = tt {
                                reduce_single_token(subtree)
                            } else {
                                tt
                            }
                        }
                        Fragment::Ast(_) => return Ok(fragment),
                    }
                }
            }
            crate::Leaf::Literal(l) => tt::Leaf::from(tt::Literal { text: l.text.clone() }).into(),
        },
    };
    Ok(Fragment::Tokens(res))
}

#[cfg(test)]
mod tests {
    use ra_syntax::{ast, AstNode};

    use super::*;
    use crate::ast_to_token_tree;

    #[test]
    fn test_expand_rule() {
        // FIXME: The missing $var check should be in parsing phase
        // assert_err(
        //     "($i:ident) => ($j)",
        //     "foo!{a}",
        //     ExpandError::BindingError(String::from("could not find binding `j`")),
        // );

        assert_err(
            "($($i:ident);*) => ($i)",
            "foo!{a}",
            ExpandError::BindingError(String::from(
                "expected simple binding, found nested binding `i`",
            )),
        );

        assert_err("($i) => ($i)", "foo!{a}", ExpandError::UnexpectedToken);
        assert_err("($i:) => ($i)", "foo!{a}", ExpandError::UnexpectedToken);

        // FIXME:
        // Add an err test case for ($($i:ident)) => ($())
    }

    fn assert_err(macro_body: &str, invocation: &str, err: ExpandError) {
        assert_eq!(expand_first(&create_rules(&format_macro(macro_body)), invocation), Err(err));
    }

    fn format_macro(macro_body: &str) -> String {
        format!(
            "
        macro_rules! foo {{
            {}
        }}
",
            macro_body
        )
    }

    fn create_rules(macro_definition: &str) -> crate::MacroRules {
        let source_file = ast::SourceFile::parse(macro_definition).ok().unwrap();
        let macro_definition =
            source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

        let (definition_tt, _) =
            ast_to_token_tree(&macro_definition.token_tree().unwrap()).unwrap();
        crate::MacroRules::parse(&definition_tt).unwrap()
    }

    fn expand_first(
        rules: &crate::MacroRules,
        invocation: &str,
    ) -> Result<tt::Subtree, ExpandError> {
        let source_file = ast::SourceFile::parse(invocation).ok().unwrap();
        let macro_invocation =
            source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

        let (invocation_tt, _) =
            ast_to_token_tree(&macro_invocation.token_tree().unwrap()).unwrap();

        expand_rule(&rules.rules[0], &invocation_tt)
    }
}

fn push_fragment(buf: &mut Vec<tt::TokenTree>, fragment: Fragment) {
    match fragment {
        Fragment::Tokens(tt::TokenTree::Subtree(tt)) => push_subtree(buf, tt),
        Fragment::Tokens(tt) | Fragment::Ast(tt) => buf.push(tt),
    }
}

fn push_subtree(buf: &mut Vec<tt::TokenTree>, tt: tt::Subtree) {
    match tt.delimiter {
        tt::Delimiter::None => buf.extend(tt.token_trees),
        _ => buf.push(tt.into()),
    }
}
