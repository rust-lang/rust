use ra_syntax::SmolStr;

enum TokenTree {
    Leaf(Leaf),
    Subtree(Subtree),
}

enum Leaf {
    Literal(Literal),
    Punct(Punct),
    Ident(Ident),
}

struct Subtree {
    delimiter: Delimiter,
    token_trees: Vec<TokenTree>,
}

enum Delimiter {
    Parenthesis,
    Brace,
    Bracket,
    None,
}

struct Literal {
    text: SmolStr,
}

struct Punct {
    char: char,
}

struct Ident {
    text: SmolStr,
}
