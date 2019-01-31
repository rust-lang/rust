use ra_syntax::SmolStr;

use crate::macros::tt;

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

#[derive(Debug)]
enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
    Var(Var),
}

#[derive(Debug)]
struct Subtree {
    delimiter: Delimiter,
    token_trees: Vec<TokenTree>,
}

#[derive(Debug)]
enum Delimiter {
    Parenthesis,
    Brace,
    Bracket,
    None,
}

#[derive(Debug)]
struct Repeat {
    subtree: Subtree,
    kind: RepeatKind,
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
    p.eat_punct('=')?;
    p.eat_punct('>')?;
    let rhs = parse_subtree(p.eat_subtree()?)?;
    Some(Rule { lhs, rhs })
}

fn parse_subtree(tt: &tt::Subtree) -> Option<Subtree> {
    Some(Subtree {
        token_trees: Vec::new(),
        delimiter: Delimiter::None,
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

    fn bump(&mut self) {
        self.pos += 1;
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
    fn eat_punct(&mut self, char: char) -> Option<()> {
        match self.current()? {
            tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: c })) if *c == char => {
                self.bump();
                Some(())
            }
            _ => None,
        }
    }
}
