use super::*;

fn same(fmt: &'static str, p: &[Piece<'static>]) {
    let parser = Parser::new(fmt, None, vec![], false);
    assert!(parser.collect::<Vec<Piece<'static>>>() == p);
}

fn fmtdflt() -> FormatSpec<'static> {
    return FormatSpec {
        fill: None,
        align: AlignUnknown,
        flags: 0,
        precision: CountImplied,
        width: CountImplied,
        ty: "",
    };
}

fn musterr(s: &str) {
    let mut p = Parser::new(s, None, vec![], false);
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
    same("{}",
         &[NextArgument(Argument {
               position: ArgumentImplicitlyIs(0),
               format: fmtdflt(),
           })]);
}
#[test]
fn format_position() {
    same("{3}",
         &[NextArgument(Argument {
               position: ArgumentIs(3),
               format: fmtdflt(),
           })]);
}
#[test]
fn format_position_nothing_else() {
    same("{3:}",
         &[NextArgument(Argument {
               position: ArgumentIs(3),
               format: fmtdflt(),
           })]);
}
#[test]
fn format_type() {
    same("{3:a}",
         &[NextArgument(Argument {
               position: ArgumentIs(3),
               format: FormatSpec {
                   fill: None,
                   align: AlignUnknown,
                   flags: 0,
                   precision: CountImplied,
                   width: CountImplied,
                   ty: "a",
               },
           })]);
}
#[test]
fn format_align_fill() {
    same("{3:>}",
         &[NextArgument(Argument {
               position: ArgumentIs(3),
               format: FormatSpec {
                   fill: None,
                   align: AlignRight,
                   flags: 0,
                   precision: CountImplied,
                   width: CountImplied,
                   ty: "",
               },
           })]);
    same("{3:0<}",
         &[NextArgument(Argument {
               position: ArgumentIs(3),
               format: FormatSpec {
                   fill: Some('0'),
                   align: AlignLeft,
                   flags: 0,
                   precision: CountImplied,
                   width: CountImplied,
                   ty: "",
               },
           })]);
    same("{3:*<abcd}",
         &[NextArgument(Argument {
               position: ArgumentIs(3),
               format: FormatSpec {
                   fill: Some('*'),
                   align: AlignLeft,
                   flags: 0,
                   precision: CountImplied,
                   width: CountImplied,
                   ty: "abcd",
               },
           })]);
}
#[test]
fn format_counts() {
    use syntax_pos::{GLOBALS, Globals, edition};
    GLOBALS.set(&Globals::new(edition::DEFAULT_EDITION), || {
    same("{:10s}",
         &[NextArgument(Argument {
               position: ArgumentImplicitlyIs(0),
               format: FormatSpec {
                   fill: None,
                   align: AlignUnknown,
                   flags: 0,
                   precision: CountImplied,
                   width: CountIs(10),
                   ty: "s",
               },
           })]);
    same("{:10$.10s}",
         &[NextArgument(Argument {
               position: ArgumentImplicitlyIs(0),
               format: FormatSpec {
                   fill: None,
                   align: AlignUnknown,
                   flags: 0,
                   precision: CountIs(10),
                   width: CountIsParam(10),
                   ty: "s",
               },
           })]);
    same("{:.*s}",
         &[NextArgument(Argument {
               position: ArgumentImplicitlyIs(1),
               format: FormatSpec {
                   fill: None,
                   align: AlignUnknown,
                   flags: 0,
                   precision: CountIsParam(0),
                   width: CountImplied,
                   ty: "s",
               },
           })]);
    same("{:.10$s}",
         &[NextArgument(Argument {
               position: ArgumentImplicitlyIs(0),
               format: FormatSpec {
                   fill: None,
                   align: AlignUnknown,
                   flags: 0,
                   precision: CountIsParam(10),
                   width: CountImplied,
                   ty: "s",
               },
           })]);
    same("{:a$.b$s}",
         &[NextArgument(Argument {
               position: ArgumentImplicitlyIs(0),
               format: FormatSpec {
                   fill: None,
                   align: AlignUnknown,
                   flags: 0,
                   precision: CountIsName(Symbol::intern("b")),
                   width: CountIsName(Symbol::intern("a")),
                   ty: "s",
               },
           })]);
    });
}
#[test]
fn format_flags() {
    same("{:-}",
         &[NextArgument(Argument {
               position: ArgumentImplicitlyIs(0),
               format: FormatSpec {
                   fill: None,
                   align: AlignUnknown,
                   flags: (1 << FlagSignMinus as u32),
                   precision: CountImplied,
                   width: CountImplied,
                   ty: "",
               },
           })]);
    same("{:+#}",
         &[NextArgument(Argument {
               position: ArgumentImplicitlyIs(0),
               format: FormatSpec {
                   fill: None,
                   align: AlignUnknown,
                   flags: (1 << FlagSignPlus as u32) | (1 << FlagAlternate as u32),
                   precision: CountImplied,
                   width: CountImplied,
                   ty: "",
               },
           })]);
}
#[test]
fn format_mixture() {
    same("abcd {3:a} efg",
         &[String("abcd "),
           NextArgument(Argument {
               position: ArgumentIs(3),
               format: FormatSpec {
                   fill: None,
                   align: AlignUnknown,
                   flags: 0,
                   precision: CountImplied,
                   width: CountImplied,
                   ty: "a",
               },
           }),
           String(" efg")]);
}
