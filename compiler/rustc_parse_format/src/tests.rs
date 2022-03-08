use super::*;

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
    same("{}", &[NextArgument(Argument { position: ArgumentImplicitlyIs(0), format: fmtdflt() })]);
}
#[test]
fn format_position() {
    same("{3}", &[NextArgument(Argument { position: ArgumentIs(3), format: fmtdflt() })]);
}
#[test]
fn format_position_nothing_else() {
    same("{3:}", &[NextArgument(Argument { position: ArgumentIs(3), format: fmtdflt() })]);
}
#[test]
fn format_type() {
    same(
        "{3:x}",
        &[NextArgument(Argument {
            position: ArgumentIs(3),
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
    rustc_span::create_default_session_globals_then(|| {
        same(
            "{:10x}",
            &[NextArgument(Argument {
                position: ArgumentImplicitlyIs(0),
                format: FormatSpec {
                    fill: None,
                    align: AlignUnknown,
                    flags: 0,
                    precision: CountImplied,
                    width: CountIs(10),
                    precision_span: None,
                    width_span: None,
                    ty: "x",
                    ty_span: None,
                },
            })],
        );
        same(
            "{:10$.10x}",
            &[NextArgument(Argument {
                position: ArgumentImplicitlyIs(0),
                format: FormatSpec {
                    fill: None,
                    align: AlignUnknown,
                    flags: 0,
                    precision: CountIs(10),
                    width: CountIsParam(10),
                    precision_span: None,
                    width_span: Some(InnerSpan::new(3, 6)),
                    ty: "x",
                    ty_span: None,
                },
            })],
        );
        same(
            "{:.*x}",
            &[NextArgument(Argument {
                position: ArgumentImplicitlyIs(1),
                format: FormatSpec {
                    fill: None,
                    align: AlignUnknown,
                    flags: 0,
                    precision: CountIsParam(0),
                    width: CountImplied,
                    precision_span: Some(InnerSpan::new(3, 5)),
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
                format: FormatSpec {
                    fill: None,
                    align: AlignUnknown,
                    flags: 0,
                    precision: CountIsName(Symbol::intern("b"), InnerSpan::new(6, 7)),
                    width: CountIsName(Symbol::intern("a"), InnerSpan::new(4, 4)),
                    precision_span: None,
                    width_span: None,
                    ty: "?",
                    ty_span: None,
                },
            })],
        );
    });
}
#[test]
fn format_flags() {
    same(
        "{:-}",
        &[NextArgument(Argument {
            position: ArgumentImplicitlyIs(0),
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
