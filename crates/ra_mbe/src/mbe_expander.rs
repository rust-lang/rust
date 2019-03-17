/// This module takes a (parsed) definition of `macro_rules` invocation, a
/// `tt::TokenTree` representing an argument of macro invocation, and produces a
/// `tt::TokenTree` for the result of the expansion.
use rustc_hash::FxHashMap;
use ra_syntax::SmolStr;
use tt::TokenId;

use crate::ExpandError;
use crate::tt_cursor::TtCursor;

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
    expand_subtree(&rule.rhs, &bindings, &mut Vec::new())
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
    Simple(tt::TokenTree),
    Nested(Vec<Binding>),
}

impl Bindings {
    fn get(&self, name: &SmolStr, nesting: &[usize]) -> Result<&tt::TokenTree, ExpandError> {
        let mut b = self
            .inner
            .get(name)
            .ok_or(ExpandError::BindingError(format!("could not find binding `{}`", name)))?;
        for &idx in nesting.iter() {
            b = match b {
                Binding::Simple(_) => break,
                Binding::Nested(bs) => bs.get(idx).ok_or(ExpandError::BindingError(format!(
                    "could not find nested binding `{}`",
                    name
                )))?,
            };
        }
        match b {
            Binding::Simple(it) => Ok(it),
            Binding::Nested(_) => Err(ExpandError::BindingError(format!(
                "expected simple binding, found nested binding `{}`",
                name
            ))),
        }
    }

    fn push_nested(&mut self, nested: Bindings) -> Result<(), ExpandError> {
        for (key, value) in nested.inner {
            if !self.inner.contains_key(&key) {
                self.inner.insert(key.clone(), Binding::Nested(Vec::new()));
            }
            match self.inner.get_mut(&key) {
                Some(Binding::Nested(it)) => it.push(value),
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
}

fn match_lhs(pattern: &crate::Subtree, input: &mut TtCursor) -> Result<Bindings, ExpandError> {
    let mut res = Bindings::default();
    for pat in pattern.token_trees.iter() {
        match pat {
            crate::TokenTree::Leaf(leaf) => match leaf {
                crate::Leaf::Var(crate::Var { text, kind }) => {
                    let kind = kind.clone().ok_or(ExpandError::UnexpectedToken)?;
                    match kind.as_str() {
                        "ident" => {
                            let ident =
                                input.eat_ident().ok_or(ExpandError::UnexpectedToken)?.clone();
                            res.inner.insert(
                                text.clone(),
                                Binding::Simple(tt::Leaf::from(ident).into()),
                            );
                        }
                        _ => return Err(ExpandError::UnexpectedToken),
                    }
                }
                crate::Leaf::Punct(punct) => {
                    if input.eat_punct() != Some(punct) {
                        return Err(ExpandError::UnexpectedToken);
                    }
                }
                crate::Leaf::Ident(ident) => {
                    if input.eat_ident().map(|i| &i.text) != Some(&ident.text) {
                        return Err(ExpandError::UnexpectedToken);
                    }
                }
                _ => return Err(ExpandError::UnexpectedToken),
            },
            crate::TokenTree::Repeat(crate::Repeat { subtree, kind: _, separator }) => {
                // Dirty hack to make macro-expansion terminate.
                // This should be replaced by a propper macro-by-example implementation
                let mut limit = 128;
                while let Ok(nested) = match_lhs(subtree, input) {
                    limit -= 1;
                    if limit == 0 {
                        break;
                    }
                    res.push_nested(nested)?;
                    if let Some(separator) = *separator {
                        if !input.is_eof() {
                            if input.eat_punct().map(|p| p.char) != Some(separator) {
                                return Err(ExpandError::UnexpectedToken);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(res)
}

fn expand_subtree(
    template: &crate::Subtree,
    bindings: &Bindings,
    nesting: &mut Vec<usize>,
) -> Result<tt::Subtree, ExpandError> {
    let token_trees = template
        .token_trees
        .iter()
        .map(|it| expand_tt(it, bindings, nesting))
        .collect::<Result<Vec<_>, ExpandError>>()?;

    Ok(tt::Subtree { token_trees, delimiter: template.delimiter })
}

fn expand_tt(
    template: &crate::TokenTree,
    bindings: &Bindings,
    nesting: &mut Vec<usize>,
) -> Result<tt::TokenTree, ExpandError> {
    let res: tt::TokenTree = match template {
        crate::TokenTree::Subtree(subtree) => expand_subtree(subtree, bindings, nesting)?.into(),
        crate::TokenTree::Repeat(repeat) => {
            let mut token_trees = Vec::new();
            nesting.push(0);
            // Dirty hack to make macro-expansion terminate.
            // This should be replaced by a propper macro-by-example implementation
            let mut limit = 128;
            while let Ok(t) = expand_subtree(&repeat.subtree, bindings, nesting) {
                limit -= 1;
                if limit == 0 {
                    break;
                }
                let idx = nesting.pop().unwrap();
                nesting.push(idx + 1);
                token_trees.push(t.into())
            }
            nesting.pop().unwrap();
            tt::Subtree { token_trees, delimiter: tt::Delimiter::None }.into()
        }
        crate::TokenTree::Leaf(leaf) => match leaf {
            crate::Leaf::Ident(ident) => {
                tt::Leaf::from(tt::Ident { text: ident.text.clone(), id: TokenId::unspecified() })
                    .into()
            }
            crate::Leaf::Punct(punct) => tt::Leaf::from(punct.clone()).into(),
            crate::Leaf::Var(v) => bindings.get(&v.text, nesting)?.clone(),
            crate::Leaf::Literal(l) => tt::Leaf::from(tt::Literal { text: l.text.clone() }).into(),
        },
    };
    Ok(res)
}

#[cfg(test)]
mod tests {
    use ra_syntax::{ast, AstNode};

    use super::*;
    use crate::ast_to_token_tree;

    #[test]
    fn test_expand_rule() {
        assert_err(
            "($i:ident) => ($j)",
            "foo!{a}",
            ExpandError::BindingError(String::from("could not find binding `j`")),
        );

        assert_err(
            "($($i:ident);*) => ($i)",
            "foo!{a}",
            ExpandError::BindingError(String::from(
                "expected simple binding, found nested binding `i`",
            )),
        );

        assert_err("($i) => ($i)", "foo!{a}", ExpandError::UnexpectedToken);
        assert_err("($i:) => ($i)", "foo!{a}", ExpandError::UnexpectedToken);
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
        let source_file = ast::SourceFile::parse(macro_definition);
        let macro_definition =
            source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

        let (definition_tt, _) = ast_to_token_tree(macro_definition.token_tree().unwrap()).unwrap();
        crate::MacroRules::parse(&definition_tt).unwrap()
    }

    fn expand_first(
        rules: &crate::MacroRules,
        invocation: &str,
    ) -> Result<tt::Subtree, ExpandError> {
        let source_file = ast::SourceFile::parse(invocation);
        let macro_invocation =
            source_file.syntax().descendants().find_map(ast::MacroCall::cast).unwrap();

        let (invocation_tt, _) = ast_to_token_tree(macro_invocation.token_tree().unwrap()).unwrap();

        expand_rule(&rules.rules[0], &invocation_tt)
    }
}
