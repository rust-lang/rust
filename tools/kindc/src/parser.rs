use crate::ast::*;
use std::iter::Peekable;
use std::str::Chars;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    Kind,
    Struct,
    Enum,
    Version,
    Ident(String),
    Int(u32),
    LBrace,
    RBrace,
    LParen,
    RParen,
    LAngle,
    RAngle,
    Comma,
    Colon,
    Eq,
    Dot,
    DocComment(String),
    Eof,
}

pub struct Lexer<'a> {
    input: &'a str,
    chars: Peekable<Chars<'a>>,
    line: usize,
    col: usize,
    file: String,
}

impl<'a> Lexer<'a> {
    pub fn new(file: String, input: &'a str) -> Self {
        Self {
            input,
            chars: input.chars().peekable(),
            line: 1,
            col: 1,
            file,
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.chars.peek().cloned()
    }

    fn next(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.next();
            } else if c == '#' {
                while let Some(c) = self.next() {
                    if c == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    pub fn next_token(&mut self) -> Result<(Token, Span), String> {
        self.skip_whitespace();

        let span = Span {
            file: self.file.clone(),
            line: self.line,
            col: self.col,
        };

        let c = match self.next() {
            Some(c) => c,
            None => return Ok((Token::Eof, span)),
        };

        let token = match c {
            '{' => Token::LBrace,
            '}' => Token::RBrace,
            '(' => Token::LParen,
            ')' => Token::RParen,
            '<' => Token::LAngle,
            '>' => Token::RAngle,
            ',' => Token::Comma,
            ':' => Token::Colon,
            '=' => Token::Eq,
            '.' => Token::Dot,
            '/' if self.peek() == Some('/') => {
                self.next(); // /
                if self.peek() == Some('/') {
                    self.next(); // /
                    let mut comment = String::new();
                    while let Some(c) = self.next() {
                        if c == '\n' {
                            break;
                        }
                        comment.push(c);
                    }
                    Token::DocComment(comment.trim().to_string())
                } else {
                    // Regular // comment, skip it and get next token
                    while let Some(c) = self.next() {
                        if c == '\n' {
                            break;
                        }
                    }
                    return self.next_token();
                }
            }
            c if c.is_ascii_digit() => {
                let mut s = String::from(c);
                while let Some(peeked) = self.peek() {
                    if peeked.is_ascii_digit() {
                        s.push(self.next().unwrap());
                    } else {
                        break;
                    }
                }
                Token::Int(s.parse::<u32>().map_err(|e| e.to_string())?)
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut s = String::from(c);
                while let Some(peeked) = self.peek() {
                    if peeked.is_ascii_alphanumeric() || peeked == '_' {
                        s.push(self.next().unwrap());
                    } else {
                        break;
                    }
                }
                match s.as_str() {
                    "kind" => Token::Kind,
                    "struct" => Token::Struct,
                    "enum" => Token::Enum,
                    "version" => Token::Version,
                    _ => Token::Ident(s),
                }
            }
            _ => return Err(format!("Unexpected character: {}", c)),
        };

        Ok((token, span))
    }
}

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    curr: (Token, Span),
    peek: (Token, Span),
}

impl<'a> Parser<'a> {
    pub fn new(file: String, input: &'a str) -> Result<Self, String> {
        let mut lexer = Lexer::new(file, input);
        let curr = lexer.next_token()?;
        let peek = lexer.next_token()?;
        Ok(Self { lexer, curr, peek })
    }

    fn advance(&mut self) -> Result<(), String> {
        self.curr = std::mem::replace(&mut self.peek, self.lexer.next_token()?);
        Ok(())
    }

    fn expect(&mut self, expected: Token) -> Result<Span, String> {
        if std::mem::discriminant(&self.curr.0) == std::mem::discriminant(&expected) {
            let span = self.curr.1.clone();
            self.advance()?;
            Ok(span)
        } else {
            Err(format!(
                "Expected {:?}, found {:?} at {:?}",
                expected, self.curr.0, self.curr.1
            ))
        }
    }

    fn parse_ident(&mut self) -> Result<String, String> {
        let name = match &self.curr.0 {
            Token::Ident(s) => s.clone(),
            Token::Kind => "kind".to_string(),
            Token::Struct => "struct".to_string(),
            Token::Enum => "enum".to_string(),
            Token::Version => "version".to_string(),
            _ => return Err(format!("Expected identifier, found {:?}", self.curr.0)),
        };
        self.advance()?;
        Ok(name)
    }

    fn parse_dotted_name(&mut self) -> Result<DottedName, String> {
        let mut name = Vec::new();
        name.push(self.parse_ident()?);
        while let Token::Dot = self.curr.0 {
            self.advance()?;
            name.push(self.parse_ident()?);
        }
        Ok(name)
    }

    fn parse_type_expr(&mut self) -> Result<TypeExpr, String> {
        let span = self.curr.1.clone();
        let name = self.parse_dotted_name()?;
        let mut args = Vec::new();
        if let Token::LAngle = self.curr.0 {
            self.advance()?;
            while self.curr.0 != Token::RAngle {
                args.push(self.parse_type_expr()?);
                if let Token::Comma = self.curr.0 {
                    self.advance()?;
                } else if self.curr.0 != Token::RAngle {
                    return Err(format!("Expected ',' or '>', found {:?}", self.curr.0));
                }
            }
            self.expect(Token::RAngle)?;
        }
        Ok(TypeExpr { name, args, span })
    }

    fn parse_field_decl(&mut self) -> Result<FieldDecl, String> {
        let mut doc = None;
        while let Token::DocComment(s) = &self.curr.0 {
            let mut combined: String = doc.unwrap_or_default();
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(s);
            doc = Some(combined);
            self.advance()?;
        }

        let span = self.curr.1.clone();
        let name = self.parse_ident()?;
        self.expect(Token::Colon)?;
        let ty = self.parse_type_expr()?;
        Ok(FieldDecl {
            doc,
            name,
            ty,
            span,
        })
    }

    fn parse_variant_decl(&mut self) -> Result<VariantDecl, String> {
        let mut doc = None;
        while let Token::DocComment(s) = &self.curr.0 {
            let mut combined: String = doc.unwrap_or_default();
            if !combined.is_empty() {
                combined.push('\n');
            }
            combined.push_str(s);
            doc = Some(combined);
            self.advance()?;
        }

        let span = self.curr.1.clone();
        let name = self.parse_ident()?;

        let payload = match &self.curr.0 {
            Token::LParen => {
                self.advance()?;
                let ty = self.parse_type_expr()?;
                self.expect(Token::RParen)?;
                VariantPayload::Tuple(ty)
            }
            Token::LBrace => {
                self.advance()?;
                let mut fields = Vec::new();
                while self.curr.0 != Token::RBrace {
                    fields.push(self.parse_field_decl()?);
                    if let Token::Comma = self.curr.0 {
                        self.advance()?;
                    } else if self.curr.0 != Token::RBrace {
                        return Err(format!("Expected ',' or '}}', found {:?}", self.curr.0));
                    }
                }
                self.expect(Token::RBrace)?;
                VariantPayload::Struct(fields)
            }
            _ => VariantPayload::Unit,
        };

        Ok(VariantDecl {
            doc,
            name,
            payload,
            span,
        })
    }

    pub fn parse_file(&mut self) -> Result<File, String> {
        let mut version = None;
        let mut declarations = Vec::new();

        while self.curr.0 != Token::Eof {
            let mut doc = None;
            while let Token::DocComment(s) = &self.curr.0 {
                let mut combined: String = doc.unwrap_or_default();
                if !combined.is_empty() {
                    combined.push('\n');
                }
                combined.push_str(s);
                doc = Some(combined);
                self.advance()?;
            }

            match &self.curr.0 {
                Token::Version => {
                    self.advance()?;
                    match self.curr.0 {
                        Token::Int(v) => {
                            version = Some(v);
                            self.advance()?;
                        }
                        _ => return Err(format!("Expected version number, found {:?}", self.curr.0)),
                    }
                }
                Token::Kind => {
                    let decl_span = self.curr.1.clone();
                    self.advance()?;
                    let name = self.parse_dotted_name()?;
                    let mut type_params = Vec::new();
                    if let Token::LAngle = self.curr.0 {
                        self.advance()?;
                        while self.curr.0 != Token::RAngle {
                            match &self.curr.0 {
                                Token::Ident(s) => type_params.push(s.clone()),
                                _ => return Err(format!("Expected type parameter name, found {:?}", self.curr.0)),
                            }
                            self.advance()?;
                            if let Token::Comma = self.curr.0 {
                                self.advance()?;
                            } else if self.curr.0 != Token::RAngle {
                                return Err(format!("Expected ',' or '>', found {:?}", self.curr.0));
                            }
                        }
                        self.expect(Token::RAngle)?;
                    }

                    let mut body = None;
                    if let Token::Eq = self.curr.0 {
                        self.advance()?;
                        match &self.curr.0 {
                            Token::Struct => {
                                self.advance()?;
                                self.expect(Token::LBrace)?;
                                let mut fields = Vec::new();
                                while self.curr.0 != Token::RBrace {
                                    fields.push(self.parse_field_decl()?);
                                    if let Token::Comma = self.curr.0 {
                                        self.advance()?;
                                    } else if self.curr.0 != Token::RBrace {
                                        return Err(format!("Expected ',' or '}}', found {:?}", self.curr.0));
                                    }
                                }
                                self.expect(Token::RBrace)?;
                                body = Some(KindBody::Struct(fields));
                            }
                            Token::Enum => {
                                self.advance()?;
                                self.expect(Token::LBrace)?;
                                let mut variants = Vec::new();
                                while self.curr.0 != Token::RBrace {
                                    variants.push(self.parse_variant_decl()?);
                                    if let Token::Comma = self.curr.0 {
                                        self.advance()?;
                                    } else if self.curr.0 != Token::RBrace {
                                        return Err(format!("Expected ',' or '}}', found {:?}", self.curr.0));
                                    }
                                }
                                self.expect(Token::RBrace)?;
                                body = Some(KindBody::Enum(variants));
                            }
                            _ => {
                                body = Some(KindBody::Alias(self.parse_type_expr()?));
                            }
                        }
                    }

                    declarations.push(KindDecl {
                        doc,
                        name,
                        type_params,
                        body,
                        span: decl_span,
                    });
                }
                _ => return Err(format!("Expected 'kind' or 'version', found {:?}", self.curr.0)),
            }
        }

        Ok(File {
            version,
            declarations,
        })
    }
}
