use super::SemverParser;

#[allow(dead_code, non_camel_case_types)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Rule {
    EOI,
    range_set,
    logical_or,
    range,
    empty,
    hyphen,
    simple,
    primitive,
    primitive_op,
    partial,
    xr,
    xr_op,
    nr,
    tilde,
    caret,
    qualifier,
    parts,
    part,
    space,
}
#[allow(clippy::all)]
impl ::pest::Parser<Rule> for SemverParser {
    fn parse<'i>(
        rule: Rule,
        input: &'i str,
    ) -> ::std::result::Result<::pest::iterators::Pairs<'i, Rule>, ::pest::error::Error<Rule>> {
        mod rules {
            pub mod hidden {
                use super::super::Rule;
                #[inline]
                #[allow(dead_code, non_snake_case, unused_variables)]
                pub fn skip(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    Ok(state)
                }
            }
            pub mod visible {
                use super::super::Rule;
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn range_set(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::range_set, |state| {
                        state.sequence(|state| {
                            self::SOI(state)
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| {
                                    state.sequence(|state| {
                                        state.optional(|state| {
                                            self::space(state).and_then(|state| {
                                                state.repeat(|state| {
                                                    state.sequence(|state| {
                                                        super::hidden::skip(state)
                                                            .and_then(|state| self::space(state))
                                                    })
                                                })
                                            })
                                        })
                                    })
                                })
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| self::range(state))
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| {
                                    state.sequence(|state| {
                                        state.optional(|state| {
                                            state
                                                .sequence(|state| {
                                                    self::logical_or(state)
                                                        .and_then(|state| {
                                                            super::hidden::skip(state)
                                                        })
                                                        .and_then(|state| self::range(state))
                                                })
                                                .and_then(|state| {
                                                    state.repeat(|state| {
                                                        state.sequence(|state| {
                                                            super::hidden::skip(state).and_then(
                                                                |state| {
                                                                    state.sequence(|state| {
                                                                        self::logical_or(state)
                                                                            .and_then(|state| {
                                                                                super::hidden::skip(
                                                                                    state,
                                                                                )
                                                                            })
                                                                            .and_then(|state| {
                                                                                self::range(state)
                                                                            })
                                                                    })
                                                                },
                                                            )
                                                        })
                                                    })
                                                })
                                        })
                                    })
                                })
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| {
                                    state.sequence(|state| {
                                        state.optional(|state| {
                                            self::space(state).and_then(|state| {
                                                state.repeat(|state| {
                                                    state.sequence(|state| {
                                                        super::hidden::skip(state)
                                                            .and_then(|state| self::space(state))
                                                    })
                                                })
                                            })
                                        })
                                    })
                                })
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| self::EOI(state))
                        })
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn logical_or(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::logical_or, |state| {
                        state.sequence(|state| {
                            state
                                .sequence(|state| {
                                    state.optional(|state| {
                                        self::space(state).and_then(|state| {
                                            state.repeat(|state| {
                                                state.sequence(|state| {
                                                    super::hidden::skip(state)
                                                        .and_then(|state| self::space(state))
                                                })
                                            })
                                        })
                                    })
                                })
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| state.match_string("||"))
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| {
                                    state.sequence(|state| {
                                        state.optional(|state| {
                                            self::space(state).and_then(|state| {
                                                state.repeat(|state| {
                                                    state.sequence(|state| {
                                                        super::hidden::skip(state)
                                                            .and_then(|state| self::space(state))
                                                    })
                                                })
                                            })
                                        })
                                    })
                                })
                        })
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn range(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::range, |state| {
                        self::hyphen(state)
                            .or_else(|state| {
                                state.sequence(|state| {
                                    self::simple(state)
                                        .and_then(|state| super::hidden::skip(state))
                                        .and_then(|state| {
                                            state.sequence(|state| {
                                                state.optional(|state| {
                                                    state
                                                        .sequence(|state| {
                                                            state
                                                                .optional(|state| {
                                                                    state.match_string(",")
                                                                })
                                                                .and_then(|state| {
                                                                    super::hidden::skip(state)
                                                                })
                                                                .and_then(|state| {
                                                                    state.sequence(|state| {
                                                                        self::space(state)
                                      .and_then(|state| super::hidden::skip(state))
                                      .and_then(|state| {
                                        state.sequence(|state| {
                                          state.optional(|state| {
                                            self::space(state).and_then(|state| {
                                              state.repeat(|state| {
                                                state.sequence(|state| {
                                                  super::hidden::skip(state)
                                                    .and_then(|state| self::space(state))
                                                })
                                              })
                                            })
                                          })
                                        })
                                      })
                                                                    })
                                                                })
                                                                .and_then(|state| {
                                                                    super::hidden::skip(state)
                                                                })
                                                                .and_then(|state| {
                                                                    self::simple(state)
                                                                })
                                                        })
                                                        .and_then(|state| {
                                                            state.repeat(|state| {
                                                                state.sequence(|state| {
                                                                    super::hidden::skip(state)
                                                                        .and_then(|state| {
                                                                            state.sequence(
                                                                                |state| {
                                                                                    state
                                        .optional(|state| state.match_string(","))
                                        .and_then(|state| super::hidden::skip(state))
                                        .and_then(|state| {
                                          state.sequence(|state| {
                                            self::space(state)
                                              .and_then(|state| super::hidden::skip(state))
                                              .and_then(|state| {
                                                state.sequence(|state| {
                                                  state.optional(|state| {
                                                    self::space(state).and_then(|state| {
                                                      state.repeat(|state| {
                                                        state.sequence(|state| {
                                                          super::hidden::skip(state)
                                                            .and_then(|state| self::space(state))
                                                        })
                                                      })
                                                    })
                                                  })
                                                })
                                              })
                                          })
                                        })
                                        .and_then(|state| super::hidden::skip(state))
                                        .and_then(|state| self::simple(state))
                                                                                },
                                                                            )
                                                                        })
                                                                })
                                                            })
                                                        })
                                                })
                                            })
                                        })
                                })
                            })
                            .or_else(|state| self::empty(state))
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn empty(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::empty, |state| state.match_string(""))
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn hyphen(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::hyphen, |state| {
                        state.sequence(|state| {
                            self::partial(state)
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| {
                                    state.sequence(|state| {
                                        self::space(state)
                                            .and_then(|state| super::hidden::skip(state))
                                            .and_then(|state| {
                                                state.sequence(|state| {
                                                    state.optional(|state| {
                                                        self::space(state).and_then(|state| {
                                                            state.repeat(|state| {
                                                                state.sequence(|state| {
                                                                    super::hidden::skip(state)
                                                                        .and_then(|state| {
                                                                            self::space(state)
                                                                        })
                                                                })
                                                            })
                                                        })
                                                    })
                                                })
                                            })
                                    })
                                })
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| state.match_string("-"))
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| {
                                    state.sequence(|state| {
                                        self::space(state)
                                            .and_then(|state| super::hidden::skip(state))
                                            .and_then(|state| {
                                                state.sequence(|state| {
                                                    state.optional(|state| {
                                                        self::space(state).and_then(|state| {
                                                            state.repeat(|state| {
                                                                state.sequence(|state| {
                                                                    super::hidden::skip(state)
                                                                        .and_then(|state| {
                                                                            self::space(state)
                                                                        })
                                                                })
                                                            })
                                                        })
                                                    })
                                                })
                                            })
                                    })
                                })
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| self::partial(state))
                        })
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn simple(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::simple, |state| {
                        self::primitive(state)
                            .or_else(|state| self::partial(state))
                            .or_else(|state| self::tilde(state))
                            .or_else(|state| self::caret(state))
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn primitive(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::primitive, |state| {
                        state.sequence(|state| {
                            self::primitive_op(state)
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| {
                                    state.sequence(|state| {
                                        state.optional(|state| {
                                            self::space(state).and_then(|state| {
                                                state.repeat(|state| {
                                                    state.sequence(|state| {
                                                        super::hidden::skip(state)
                                                            .and_then(|state| self::space(state))
                                                    })
                                                })
                                            })
                                        })
                                    })
                                })
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| self::partial(state))
                        })
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn primitive_op(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::primitive_op, |state| {
                        state
                            .match_string("<=")
                            .or_else(|state| state.match_string(">="))
                            .or_else(|state| state.match_string(">"))
                            .or_else(|state| state.match_string("<"))
                            .or_else(|state| state.match_string("="))
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn partial(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::partial, |state| {
                        state.sequence(|state| {
                            self::xr(state)
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| {
                                    state.optional(|state| {
                                        state.sequence(|state| {
                                            state
                                                .match_string(".")
                                                .and_then(|state| super::hidden::skip(state))
                                                .and_then(|state| self::xr(state))
                                                .and_then(|state| super::hidden::skip(state))
                                                .and_then(|state| {
                                                    state.optional(|state| {
                                                        state.sequence(|state| {
                                                            state
                                                                .match_string(".")
                                                                .and_then(|state| {
                                                                    super::hidden::skip(state)
                                                                })
                                                                .and_then(|state| self::xr(state))
                                                                .and_then(|state| {
                                                                    super::hidden::skip(state)
                                                                })
                                                                .and_then(|state| {
                                                                    state.optional(|state| {
                                                                        self::qualifier(state)
                                                                    })
                                                                })
                                                        })
                                                    })
                                                })
                                        })
                                    })
                                })
                        })
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn xr(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::xr, |state| {
                        self::xr_op(state).or_else(|state| self::nr(state))
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn xr_op(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::xr_op, |state| {
                        state
                            .match_string("x")
                            .or_else(|state| state.match_string("X"))
                            .or_else(|state| state.match_string("*"))
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn nr(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::nr, |state| {
                        state.match_string("0").or_else(|state| {
                            state.sequence(|state| {
                                state
                                    .match_range('1'..'9')
                                    .and_then(|state| super::hidden::skip(state))
                                    .and_then(|state| {
                                        state.sequence(|state| {
                                            state.optional(|state| {
                                                state.match_range('0'..'9').and_then(|state| {
                                                    state.repeat(|state| {
                                                        state.sequence(|state| {
                                                            super::hidden::skip(state).and_then(
                                                                |state| state.match_range('0'..'9'),
                                                            )
                                                        })
                                                    })
                                                })
                                            })
                                        })
                                    })
                            })
                        })
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn tilde(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::tilde, |state| {
                        state.sequence(|state| {
                            state
                                .match_string("~>")
                                .or_else(|state| state.match_string("~"))
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| {
                                    state.sequence(|state| {
                                        state.optional(|state| {
                                            self::space(state).and_then(|state| {
                                                state.repeat(|state| {
                                                    state.sequence(|state| {
                                                        super::hidden::skip(state)
                                                            .and_then(|state| self::space(state))
                                                    })
                                                })
                                            })
                                        })
                                    })
                                })
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| self::partial(state))
                        })
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn caret(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::caret, |state| {
                        state.sequence(|state| {
                            state
                                .match_string("^")
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| {
                                    state.sequence(|state| {
                                        state.optional(|state| {
                                            self::space(state).and_then(|state| {
                                                state.repeat(|state| {
                                                    state.sequence(|state| {
                                                        super::hidden::skip(state)
                                                            .and_then(|state| self::space(state))
                                                    })
                                                })
                                            })
                                        })
                                    })
                                })
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| self::partial(state))
                        })
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn qualifier(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::qualifier, |state| {
                        state.sequence(|state| {
                            state
                                .match_string("-")
                                .or_else(|state| state.match_string("+"))
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| self::parts(state))
                        })
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn parts(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::parts, |state| {
                        state.sequence(|state| {
                            self::part(state)
                                .and_then(|state| super::hidden::skip(state))
                                .and_then(|state| {
                                    state.sequence(|state| {
                                        state.optional(|state| {
                                            state
                                                .sequence(|state| {
                                                    state
                                                        .match_string(".")
                                                        .and_then(|state| {
                                                            super::hidden::skip(state)
                                                        })
                                                        .and_then(|state| self::part(state))
                                                })
                                                .and_then(|state| {
                                                    state.repeat(|state| {
                                                        state.sequence(|state| {
                                                            super::hidden::skip(state).and_then(
                                                                |state| {
                                                                    state.sequence(|state| {
                                                                        state
                                                                            .match_string(".")
                                                                            .and_then(|state| {
                                                                                super::hidden::skip(
                                                                                    state,
                                                                                )
                                                                            })
                                                                            .and_then(|state| {
                                                                                self::part(state)
                                                                            })
                                                                    })
                                                                },
                                                            )
                                                        })
                                                    })
                                                })
                                        })
                                    })
                                })
                        })
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn part(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::part, |state| {
                        self::nr(state).or_else(|state| {
                            state.sequence(|state| {
                                state
                                    .match_string("-")
                                    .or_else(|state| state.match_range('0'..'9'))
                                    .or_else(|state| state.match_range('A'..'Z'))
                                    .or_else(|state| state.match_range('a'..'z'))
                                    .and_then(|state| super::hidden::skip(state))
                                    .and_then(|state| {
                                        state.sequence(|state| {
                                            state.optional(|state| {
                                                state
                                                    .match_string("-")
                                                    .or_else(|state| state.match_range('0'..'9'))
                                                    .or_else(|state| state.match_range('A'..'Z'))
                                                    .or_else(|state| state.match_range('a'..'z'))
                                                    .and_then(|state| {
                                                        state.repeat(|state| {
                                                            state.sequence(|state| {
                                                                super::hidden::skip(state).and_then(
                                                                    |state| {
                                                                        state
                                                                            .match_string("-")
                                                                            .or_else(|state| {
                                                                                state.match_range(
                                                                                    '0'..'9',
                                                                                )
                                                                            })
                                                                            .or_else(|state| {
                                                                                state.match_range(
                                                                                    'A'..'Z',
                                                                                )
                                                                            })
                                                                            .or_else(|state| {
                                                                                state.match_range(
                                                                                    'a'..'z',
                                                                                )
                                                                            })
                                                                    },
                                                                )
                                                            })
                                                        })
                                                    })
                                            })
                                        })
                                    })
                            })
                        })
                    })
                }
                #[inline]
                #[allow(non_snake_case, unused_variables)]
                pub fn space(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state
                        .match_string(" ")
                        .or_else(|state| state.match_string("\t"))
                }
                #[inline]
                #[allow(dead_code, non_snake_case, unused_variables)]
                pub fn EOI(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.rule(Rule::EOI, |state| state.end_of_input())
                }
                #[inline]
                #[allow(dead_code, non_snake_case, unused_variables)]
                pub fn SOI(
                    state: Box<::pest::ParserState<Rule>>,
                ) -> ::pest::ParseResult<Box<::pest::ParserState<Rule>>> {
                    state.start_of_input()
                }
            }
            pub use self::visible::*;
        }
        ::pest::state(input, |state| match rule {
            Rule::range_set => rules::range_set(state),
            Rule::logical_or => rules::logical_or(state),
            Rule::range => rules::range(state),
            Rule::empty => rules::empty(state),
            Rule::hyphen => rules::hyphen(state),
            Rule::simple => rules::simple(state),
            Rule::primitive => rules::primitive(state),
            Rule::primitive_op => rules::primitive_op(state),
            Rule::partial => rules::partial(state),
            Rule::xr => rules::xr(state),
            Rule::xr_op => rules::xr_op(state),
            Rule::nr => rules::nr(state),
            Rule::tilde => rules::tilde(state),
            Rule::caret => rules::caret(state),
            Rule::qualifier => rules::qualifier(state),
            Rule::parts => rules::parts(state),
            Rule::part => rules::part(state),
            Rule::space => rules::space(state),
            Rule::EOI => rules::EOI(state),
        })
    }
}
