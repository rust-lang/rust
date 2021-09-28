use super::{parse_next_substitution as pns, Substitution as S};

macro_rules! assert_eq_pnsat {
    ($lhs:expr, $rhs:expr) => {
        assert_eq!(
            pns($lhs).and_then(|(f, _)| f.translate().ok()),
            $rhs.map(<String as From<&str>>::from)
        )
    };
}

#[test]
fn test_escape() {
    assert_eq!(pns("has no escapes"), None);
    assert_eq!(pns("has no escapes, either $"), None);
    assert_eq!(pns("*so* has a $$ escape"), Some((S::Escape((11, 13)), " escape")));
    assert_eq!(pns("$$ leading escape"), Some((S::Escape((0, 2)), " leading escape")));
    assert_eq!(pns("trailing escape $$"), Some((S::Escape((16, 18)), "")));
}

#[test]
fn test_parse() {
    macro_rules! assert_pns_eq_sub {
        ($in_:expr, $kind:ident($arg:expr, $pos:expr)) => {
            assert_eq!(pns(concat!($in_, "!")), Some((S::$kind($arg.into(), $pos), "!")))
        };
    }

    assert_pns_eq_sub!("$0", Ordinal(0, (0, 2)));
    assert_pns_eq_sub!("$1", Ordinal(1, (0, 2)));
    assert_pns_eq_sub!("$9", Ordinal(9, (0, 2)));
    assert_pns_eq_sub!("$N", Name("N", (0, 2)));
    assert_pns_eq_sub!("$NAME", Name("NAME", (0, 5)));
}

#[test]
fn test_iter() {
    use super::iter_subs;
    let s = "The $0'th word $$ is: `$WORD` $!\n";
    let subs: Vec<_> = iter_subs(s, 0).map(|sub| sub.translate().ok()).collect();
    assert_eq!(
        subs.iter().map(|ms| ms.as_ref().map(|s| &s[..])).collect::<Vec<_>>(),
        vec![Some("{0}"), None, Some("{WORD}")]
    );
}

#[test]
fn test_translation() {
    assert_eq_pnsat!("$0", Some("{0}"));
    assert_eq_pnsat!("$9", Some("{9}"));
    assert_eq_pnsat!("$1", Some("{1}"));
    assert_eq_pnsat!("$10", Some("{1}"));
    assert_eq_pnsat!("$stuff", Some("{stuff}"));
    assert_eq_pnsat!("$NAME", Some("{NAME}"));
    assert_eq_pnsat!("$PREFIX/bin", Some("{PREFIX}"));
}
