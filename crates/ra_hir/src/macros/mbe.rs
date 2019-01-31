use ra_syntax::SmolStr;

use crate::macros::tt::{self, Delimiter};

#[derive(Debug)]
pub(crate) struct MacroRules {
    rules: Vec<Rule>,
}

#[derive(Debug)]
struct Rule {
    lhs: Subtree,
    rhs: Subtree,
}

#[derive(Debug)]
enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
    Repeat(Repeat),
}
impl_froms!(TokenTree: Leaf, Subtree, Repeat);

#[derive(Debug)]
enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
    Var(Var),
}
impl_froms!(Leaf: Literal, Punct, Ident, Var);

#[derive(Debug)]
struct Subtree {
    delimiter: Delimiter,
    token_trees: Vec<TokenTree>,
}

#[derive(Debug)]
struct Repeat {
    subtree: Subtree,
    kind: RepeatKind,
    separator: Option<Punct>,
}

#[derive(Debug)]
enum RepeatKind {
    ZeroOrMore,
    OneOrMore,
    ZeroOrOne,
}

#[derive(Debug)]
struct Literal {
    text: SmolStr,
}

#[derive(Debug)]
struct Punct {
    char: char,
}

#[derive(Debug)]
struct Ident {
    text: SmolStr,
}

#[derive(Debug)]
struct Var {
    text: SmolStr,
    kind: Option<SmolStr>,
}

pub(crate) fn parse(tt: &tt::Subtree) -> Option<MacroRules> {
    let mut parser = RulesParser::new(tt);
    let mut rules = Vec::new();
    while !parser.is_eof() {
        rules.push(parse_rule(&mut parser)?)
    }
    Some(MacroRules { rules })
}

fn parse_rule(p: &mut RulesParser) -> Option<Rule> {
    let lhs = parse_subtree(p.eat_subtree()?)?;
    p.expect_char('=')?;
    p.expect_char('>')?;
    let rhs = parse_subtree(p.eat_subtree()?)?;
    Some(Rule { lhs, rhs })
}

fn parse_subtree(tt: &tt::Subtree) -> Option<Subtree> {
    let mut token_trees = Vec::new();
    let mut p = RulesParser::new(tt);
    while let Some(tt) = p.eat() {
        let child: TokenTree = match tt {
            tt::TokenTree::Leaf(leaf) => match leaf {
                tt::Leaf::Punct(tt::Punct { char: '$' }) => {
                    if p.at_ident().is_some() {
                        Leaf::from(parse_var(&mut p)?).into()
                    } else {
                        parse_repeat(&mut p)?.into()
                    }
                }
                tt::Leaf::Punct(tt::Punct { char }) => Leaf::from(Punct { char: *char }).into(),
                tt::Leaf::Ident(tt::Ident { text }) => {
                    Leaf::from(Ident { text: text.clone() }).into()
                }
                tt::Leaf::Literal(tt::Literal { text }) => {
                    Leaf::from(Literal { text: text.clone() }).into()
                }
            },
            tt::TokenTree::Subtree(subtree) => parse_subtree(subtree)?.into(),
        };
        token_trees.push(child);
    }
    Some(Subtree {
        token_trees,
        delimiter: tt.delimiter,
    })
}

fn parse_var(p: &mut RulesParser) -> Option<Var> {
    let ident = p.eat_ident().unwrap();
    let text = ident.text.clone();
    let kind = if p.at_char(':') {
        p.bump();
        if let Some(ident) = p.eat_ident() {
            Some(ident.text.clone())
        } else {
            // ugly as hell :(
            p.pos -= 1;
            None
        }
    } else {
        None
    };
    Some(Var { text, kind })
}

fn parse_repeat(p: &mut RulesParser) -> Option<Repeat> {
    let subtree = p.eat_subtree().unwrap();
    let subtree = parse_subtree(subtree)?;
    let mut sep = p.eat_punct()?;
    let (separator, rep) = match sep.char {
        '*' | '+' | '?' => (None, sep.char),
        char => (Some(Punct { char }), p.eat_punct()?.char),
    };

    let kind = match rep {
        '*' => RepeatKind::ZeroOrMore,
        '+' => RepeatKind::OneOrMore,
        '?' => RepeatKind::ZeroOrMore,
        _ => return None,
    };
    p.bump();
    Some(Repeat {
        subtree,
        kind,
        separator,
    })
}

struct RulesParser<'a> {
    subtree: &'a tt::Subtree,
    pos: usize,
}

impl<'a> RulesParser<'a> {
    fn new(subtree: &'a tt::Subtree) -> RulesParser<'a> {
        RulesParser { subtree, pos: 0 }
    }

    fn is_eof(&self) -> bool {
        self.pos == self.subtree.token_trees.len()
    }

    fn current(&self) -> Option<&'a tt::TokenTree> {
        self.subtree.token_trees.get(self.pos)
    }

    fn at_punct(&self) -> Option<&'a tt::Punct> {
        match self.current() {
            Some(tt::TokenTree::Leaf(tt::Leaf::Punct(it))) => Some(it),
            _ => None,
        }
    }

    fn at_char(&self, char: char) -> bool {
        match self.at_punct() {
            Some(tt::Punct { char: c }) if *c == char => true,
            _ => false,
        }
    }

    fn at_ident(&mut self) -> Option<&'a tt::Ident> {
        match self.current() {
            Some(tt::TokenTree::Leaf(tt::Leaf::Ident(i))) => Some(i),
            _ => None,
        }
    }

    fn bump(&mut self) {
        self.pos += 1;
    }

    fn eat(&mut self) -> Option<&'a tt::TokenTree> {
        match self.current() {
            Some(it) => {
                self.bump();
                Some(it)
            }
            None => None,
        }
    }

    fn eat_subtree(&mut self) -> Option<&'a tt::Subtree> {
        match self.current()? {
            tt::TokenTree::Subtree(sub) => {
                self.bump();
                Some(sub)
            }
            _ => return None,
        }
    }

    fn eat_punct(&mut self) -> Option<&'a tt::Punct> {
        if let Some(it) = self.at_punct() {
            self.bump();
            return Some(it);
        }
        None
    }

    fn eat_ident(&mut self) -> Option<&'a tt::Ident> {
        if let Some(i) = self.at_ident() {
            self.bump();
            return Some(i);
        }
        None
    }

    fn expect_char(&mut self, char: char) -> Option<()> {
        if self.at_char(char) {
            self.bump();
            return Some(());
        }
        None
    }
}
