use super::{
    Format as F,
    Num as N,
    Substitution as S,
    iter_subs,
    parse_next_substitution as pns,
};

macro_rules! assert_eq_pnsat {
    ($lhs:expr, $rhs:expr) => {
        assert_eq!(
            pns($lhs).and_then(|(s, _)| s.translate()),
            $rhs.map(<String as From<&str>>::from)
        )
    };
}

#[test]
fn test_escape() {
    assert_eq!(pns("has no escapes"), None);
    assert_eq!(pns("has no escapes, either %"), None);
    assert_eq!(pns("*so* has a %% escape"), Some((S::Escape," escape")));
    assert_eq!(pns("%% leading escape"), Some((S::Escape, " leading escape")));
    assert_eq!(pns("trailing escape %%"), Some((S::Escape, "")));
}

#[test]
fn test_parse() {
    macro_rules! assert_pns_eq_sub {
        ($in_:expr, {
            $param:expr, $flags:expr,
            $width:expr, $prec:expr, $len:expr, $type_:expr,
            $pos:expr,
        }) => {
            assert_eq!(
                pns(concat!($in_, "!")),
                Some((
                    S::Format(F {
                        span: $in_,
                        parameter: $param,
                        flags: $flags,
                        width: $width,
                        precision: $prec,
                        length: $len,
                        type_: $type_,
                        position: syntax_pos::InnerSpan::new($pos.0, $pos.1),
                    }),
                    "!"
                ))
            )
        };
    }

    assert_pns_eq_sub!("%!",
        { None, "", None, None, None, "!", (0, 2), });
    assert_pns_eq_sub!("%c",
        { None, "", None, None, None, "c", (0, 2), });
    assert_pns_eq_sub!("%s",
        { None, "", None, None, None, "s", (0, 2), });
    assert_pns_eq_sub!("%06d",
        { None, "0", Some(N::Num(6)), None, None, "d", (0, 4), });
    assert_pns_eq_sub!("%4.2f",
        { None, "", Some(N::Num(4)), Some(N::Num(2)), None, "f", (0, 5), });
    assert_pns_eq_sub!("%#x",
        { None, "#", None, None, None, "x", (0, 3), });
    assert_pns_eq_sub!("%-10s",
        { None, "-", Some(N::Num(10)), None, None, "s", (0, 5), });
    assert_pns_eq_sub!("%*s",
        { None, "", Some(N::Next), None, None, "s", (0, 3), });
    assert_pns_eq_sub!("%-10.*s",
        { None, "-", Some(N::Num(10)), Some(N::Next), None, "s", (0, 7), });
    assert_pns_eq_sub!("%-*.*s",
        { None, "-", Some(N::Next), Some(N::Next), None, "s", (0, 6), });
    assert_pns_eq_sub!("%.6i",
        { None, "", None, Some(N::Num(6)), None, "i", (0, 4), });
    assert_pns_eq_sub!("%+i",
        { None, "+", None, None, None, "i", (0, 3), });
    assert_pns_eq_sub!("%08X",
        { None, "0", Some(N::Num(8)), None, None, "X", (0, 4), });
    assert_pns_eq_sub!("%lu",
        { None, "", None, None, Some("l"), "u", (0, 3), });
    assert_pns_eq_sub!("%Iu",
        { None, "", None, None, Some("I"), "u", (0, 3), });
    assert_pns_eq_sub!("%I32u",
        { None, "", None, None, Some("I32"), "u", (0, 5), });
    assert_pns_eq_sub!("%I64u",
        { None, "", None, None, Some("I64"), "u", (0, 5), });
    assert_pns_eq_sub!("%'d",
        { None, "'", None, None, None, "d", (0, 3), });
    assert_pns_eq_sub!("%10s",
        { None, "", Some(N::Num(10)), None, None, "s", (0, 4), });
    assert_pns_eq_sub!("%-10.10s",
        { None, "-", Some(N::Num(10)), Some(N::Num(10)), None, "s", (0, 8), });
    assert_pns_eq_sub!("%1$d",
        { Some(1), "", None, None, None, "d", (0, 4), });
    assert_pns_eq_sub!("%2$.*3$d",
        { Some(2), "", None, Some(N::Arg(3)), None, "d", (0, 8), });
    assert_pns_eq_sub!("%1$*2$.*3$d",
        { Some(1), "", Some(N::Arg(2)), Some(N::Arg(3)), None, "d", (0, 11), });
    assert_pns_eq_sub!("%-8ld",
        { None, "-", Some(N::Num(8)), None, Some("l"), "d", (0, 5), });
}

#[test]
fn test_iter() {
    let s = "The %d'th word %% is: `%.*s` %!\n";
    let subs: Vec<_> = iter_subs(s, 0).map(|sub| sub.translate()).collect();
    assert_eq!(
        subs.iter().map(|ms| ms.as_ref().map(|s| &s[..])).collect::<Vec<_>>(),
        vec![Some("{}"), None, Some("{:.*}"), None]
    );
}

/// Checks that the translations are what we expect.
#[test]
fn test_translation() {
    assert_eq_pnsat!("%c", Some("{}"));
    assert_eq_pnsat!("%d", Some("{}"));
    assert_eq_pnsat!("%u", Some("{}"));
    assert_eq_pnsat!("%x", Some("{:x}"));
    assert_eq_pnsat!("%X", Some("{:X}"));
    assert_eq_pnsat!("%e", Some("{:e}"));
    assert_eq_pnsat!("%E", Some("{:E}"));
    assert_eq_pnsat!("%f", Some("{}"));
    assert_eq_pnsat!("%g", Some("{:e}"));
    assert_eq_pnsat!("%G", Some("{:E}"));
    assert_eq_pnsat!("%s", Some("{}"));
    assert_eq_pnsat!("%p", Some("{:p}"));

    assert_eq_pnsat!("%06d",        Some("{:06}"));
    assert_eq_pnsat!("%4.2f",       Some("{:4.2}"));
    assert_eq_pnsat!("%#x",         Some("{:#x}"));
    assert_eq_pnsat!("%-10s",       Some("{:<10}"));
    assert_eq_pnsat!("%*s",         None);
    assert_eq_pnsat!("%-10.*s",     Some("{:<10.*}"));
    assert_eq_pnsat!("%-*.*s",      None);
    assert_eq_pnsat!("%.6i",        Some("{:06}"));
    assert_eq_pnsat!("%+i",         Some("{:+}"));
    assert_eq_pnsat!("%08X",        Some("{:08X}"));
    assert_eq_pnsat!("%lu",         Some("{}"));
    assert_eq_pnsat!("%Iu",         Some("{}"));
    assert_eq_pnsat!("%I32u",       Some("{}"));
    assert_eq_pnsat!("%I64u",       Some("{}"));
    assert_eq_pnsat!("%'d",         None);
    assert_eq_pnsat!("%10s",        Some("{:>10}"));
    assert_eq_pnsat!("%-10.10s",    Some("{:<10.10}"));
    assert_eq_pnsat!("%1$d",        Some("{0}"));
    assert_eq_pnsat!("%2$.*3$d",    Some("{1:02$}"));
    assert_eq_pnsat!("%1$*2$.*3$s", Some("{0:>1$.2$}"));
    assert_eq_pnsat!("%-8ld",       Some("{:<8}"));
}
