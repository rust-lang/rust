use crate::{
    mbe_expander::{Binding, Bindings, Fragment},
    tt_cursor::TtCursor,
    ExpandError,
};

use ra_parser::FragmentKind::*;
use ra_syntax::SmolStr;

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

pub(super) fn match_lhs(
    pattern: &crate::Subtree,
    input: &mut TtCursor,
) -> Result<Bindings, ExpandError> {
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

fn collect_vars(subtree: &crate::Subtree) -> Vec<SmolStr> {
    let mut res = Vec::new();

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
