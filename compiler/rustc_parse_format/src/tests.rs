use super::*;

#[track_caller]
fn same(fmt: &'static str, p: &[Piece<'static>]) {
    let parser = Parser::new(fmt, None, None, false, ParseMode::Format);
    assert_eq!(parser.collect::<Vec<Piece<'static>>>(), p);
}

fn fmtdflt() -> FormatSpec<'static> {
    return FormatSpec {
        fill: None,
        align: AlignUnknown,
        flags: 0,
        precision: CountImplied,
        width: CountImplied,
        precision_span: None,
        width_span: None,
        ty: "",
        ty_span: None,
    };
}

fn musterr(s: &str) {
    let mut p = Parser::new(s, None, None, false, ParseMode::Format);
    p.next();
    assert!(!p.errors.is_empty());
}

#[test]
fn simple() {
    same("asdf", &[String("asdf")]);
    same("a{{b", &[String("a"), String("{b")]);
    same("a}}b", &[String("a"), String("}b")]);
    same("a}}", &[String("a"), String("}")]);
    same("}}", &[String("}")]);
    same("\\}}", &[String("\\"), String("}")]);
}

#[test]
fn invalid01() {
    musterr("{")
}
#[test]
fn invalid02() {
    musterr("}")
}
#[test]
fn invalid04() {
    musterr("{3a}")
}
#[test]
fn invalid05() {
    musterr("{:|}")
}
#[test]
fn invalid06() {
    musterr("{:>>>}")
}

#[test]
fn format_nothing() {
    same(
        "{}",
        &[NextArgument(Argument {
            position: ArgumentImplicitlyIs(0),
            position_span: InnerSpan { start: 2, end: 2 },
            format: fmtdflt(),
        })],
    );
}
#[test]
fn format_position() {
    same(
        "{3}",
        &[NextArgument(Argument {
            position: ArgumentIs(3),
            position_span: InnerSpan { start: 2, end: 3 },
            format: fmtdflt(),
        })],
    );
}
#[test]
fn format_position_nothing_else() {
    same(
        "{3:}",
        &[NextArgument(Argument {
            position: ArgumentIs(3),
            position_span: InnerSpan { start: 2, end: 3 },
            format: fmtdflt(),
        })],
    );
}
#[test]
fn format_named() {
    same(
        "{name}",
        &[NextArgument(Argument {
            position: ArgumentNamed("name"),
            position_span: InnerSpan { start: 2, end: 6 },
            format: fmtdflt(),
        })],
    )
}
#[test]
fn format_type() {
    same(
        "{3:x}",
        &[NextArgument(Argument {
            position: ArgumentIs(3),
            position_span: InnerSpan { start: 2, end: 3 },
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                precision_span: None,
                width_span: None,
                ty: "x",
                ty_span: None,
            },
        })],
    );
}
#[test]
fn format_align_fill() {
    same(
        "{3:>}",
        &[NextArgument(Argument {
            position: ArgumentIs(3),
            position_span: InnerSpan { start: 2, end: 3 },
            format: FormatSpec {
                fill: None,
                align: AlignRight,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                precision_span: None,
                width_span: None,
                ty: "",
                ty_span: None,
            },
        })],
    );
    same(
        "{3:0<}",
        &[NextArgument(Argument {
            position: ArgumentIs(3),
            position_span: InnerSpan { start: 2, end: 3 },
            format: FormatSpec {
                fill: Some('0'),
                align: AlignLeft,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                precision_span: None,
                width_span: None,
                ty: "",
                ty_span: None,
            },
        })],
    );
    same(
        "{3:*<abcd}",
        &[NextArgument(Argument {
            position: ArgumentIs(3),
            position_span: InnerSpan { start: 2, end: 3 },
            format: FormatSpec {
                fill: Some('*'),
                align: AlignLeft,
                flags: 0,
                precision: CountImplied,
                width: CountImplied,
                precision_span: None,
                width_span: None,
                ty: "abcd",
                ty_span: Some(InnerSpan::new(6, 10)),
            },
        })],
    );
}
#[test]
fn format_counts() {
    same(
        "{:10x}",
        &[NextArgument(Argument {
            position: ArgumentImplicitlyIs(0),
            position_span: InnerSpan { start: 2, end: 2 },
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountImplied,
                precision_span: None,
                width: CountIs(10),
                width_span: Some(InnerSpan { start: 3, end: 5 }),
                ty: "x",
                ty_span: None,
            },
        })],
    );
    same(
        "{:10$.10x}",
        &[NextArgument(Argument {
            position: ArgumentImplicitlyIs(0),
            position_span: InnerSpan { start: 2, end: 2 },
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIs(10),
                precision_span: Some(InnerSpan { start: 6, end: 9 }),
                width: CountIsParam(10),
                width_span: Some(InnerSpan { start: 3, end: 6 }),
                ty: "x",
                ty_span: None,
            },
        })],
    );
    same(
        "{1:0$.10x}",
        &[NextArgument(Argument {
            position: ArgumentIs(1),
            position_span: InnerSpan { start: 2, end: 3 },
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIs(10),
                precision_span: Some(InnerSpan { start: 6, end: 9 }),
                width: CountIsParam(0),
                width_span: Some(InnerSpan { start: 4, end: 6 }),
                ty: "x",
                ty_span: None,
            },
        })],
    );
    same(
        "{:.*x}",
        &[NextArgument(Argument {
            position: ArgumentImplicitlyIs(1),
            position_span: InnerSpan { start: 2, end: 2 },
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIsParam(0),
                precision_span: Some(InnerSpan { start: 3, end: 5 }),
                width: CountImplied,
                width_span: None,
                ty: "x",
                ty_span: None,
            },
        })],
    );
    same(
        "{:.10$x}",
        &[NextArgument(Argument {
            position: ArgumentImplicitlyIs(0),
            position_span: InnerSpan { start: 2, end: 2 },
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIsParam(10),
                width: CountImplied,
                precision_span: Some(InnerSpan::new(3, 7)),
                width_span: None,
                ty: "x",
                ty_span: None,
            },
        })],
    );
    same(
        "{:a$.b$?}",
        &[NextArgument(Argument {
            position: ArgumentImplicitlyIs(0),
            position_span: InnerSpan { start: 2, end: 2 },
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIsName("b", InnerSpan { start: 6, end: 7 }),
                precision_span: Some(InnerSpan { start: 5, end: 8 }),
                width: CountIsName("a", InnerSpan { start: 3, end: 4 }),
                width_span: Some(InnerSpan { start: 3, end: 5 }),
                ty: "?",
                ty_span: None,
            },
        })],
    );
    same(
        "{:.4}",
        &[NextArgument(Argument {
            position: ArgumentImplicitlyIs(0),
            position_span: InnerSpan { start: 2, end: 2 },
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: 0,
                precision: CountIs(4),
                precision_span: Some(InnerSpan { start: 3, end: 5 }),
                width: CountImplied,
                width_span: None,
                ty: "",
                ty_span: None,
            },
        })],
    )
}
#[test]
fn format_flags() {
    same(
        "{:-}",
        &[NextArgument(Argument {
            position: ArgumentImplicitlyIs(0),
            position_span: InnerSpan { start: 2, end: 2 },
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: (1 << FlagSignMinus as u32),
                precision: CountImplied,
                width: CountImplied,
                precision_span: None,
                width_span: None,
                ty: "",
                ty_span: None,
            },
        })],
    );
    same(
        "{:+#}",
        &[NextArgument(Argument {
            position: ArgumentImplicitlyIs(0),
            position_span: InnerSpan { start: 2, end: 2 },
            format: FormatSpec {
                fill: None,
                align: AlignUnknown,
                flags: (1 << FlagSignPlus as u32) | (1 << FlagAlternate as u32),
                precision: CountImplied,
                width: CountImplied,
                precision_span: None,
                width_span: None,
                ty: "",
                ty_span: None,
            },
        })],
    );
}
#[test]
fn format_mixture() {
    same(
        "abcd {3:x} efg",
        &[
            String("abcd "),
            NextArgument(Argument {
                position: ArgumentIs(3),
                position_span: InnerSpan { start: 7, end: 8 },
                format: FormatSpec {
                    fill: None,
                    align: AlignUnknown,
                    flags: 0,
                    precision: CountImplied,
                    width: CountImplied,
                    precision_span: None,
                    width_span: None,
                    ty: "x",
                    ty_span: None,
                },
            }),
            String(" efg"),
        ],
    );
}
#[test]
fn format_whitespace() {
    same(
        "{ }",
        &[NextArgument(Argument {
            position: ArgumentImplicitlyIs(0),
            position_span: InnerSpan { start: 2, end: 3 },
            format: fmtdflt(),
        })],
    );
    same(
        "{  }",
        &[NextArgument(Argument {
            position: ArgumentImplicitlyIs(0),
            position_span: InnerSpan { start: 2, end: 4 },
            format: fmtdflt(),
        })],
    );
}
