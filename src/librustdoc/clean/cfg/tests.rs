use super::*;

use rustc_span::DUMMY_SP;
use syntax::ast::*;
use syntax::attr;
use syntax::symbol::Symbol;
use syntax::with_default_globals;

fn word_cfg(s: &str) -> Cfg {
    Cfg::Cfg(Symbol::intern(s), None)
}

fn name_value_cfg(name: &str, value: &str) -> Cfg {
    Cfg::Cfg(Symbol::intern(name), Some(Symbol::intern(value)))
}

fn dummy_meta_item_word(name: &str) -> MetaItem {
    MetaItem {
        path: Path::from_ident(Ident::from_str(name)),
        kind: MetaItemKind::Word,
        span: DUMMY_SP,
    }
}

macro_rules! dummy_meta_item_list {
    ($name:ident, [$($list:ident),* $(,)?]) => {
        MetaItem {
            path: Path::from_ident(Ident::from_str(stringify!($name))),
            kind: MetaItemKind::List(vec![
                $(
                    NestedMetaItem::MetaItem(
                        dummy_meta_item_word(stringify!($list)),
                    ),
                )*
            ]),
            span: DUMMY_SP,
        }
    };

    ($name:ident, [$($list:expr),* $(,)?]) => {
        MetaItem {
            path: Path::from_ident(Ident::from_str(stringify!($name))),
            kind: MetaItemKind::List(vec![
                $(
                    NestedMetaItem::MetaItem($list),
                )*
            ]),
            span: DUMMY_SP,
        }
    };
}

#[test]
fn test_cfg_not() {
    with_default_globals(|| {
        assert_eq!(!Cfg::False, Cfg::True);
        assert_eq!(!Cfg::True, Cfg::False);
        assert_eq!(!word_cfg("test"), Cfg::Not(Box::new(word_cfg("test"))));
        assert_eq!(
            !Cfg::All(vec![word_cfg("a"), word_cfg("b")]),
            Cfg::Not(Box::new(Cfg::All(vec![word_cfg("a"), word_cfg("b")])))
        );
        assert_eq!(
            !Cfg::Any(vec![word_cfg("a"), word_cfg("b")]),
            Cfg::Not(Box::new(Cfg::Any(vec![word_cfg("a"), word_cfg("b")])))
        );
        assert_eq!(!Cfg::Not(Box::new(word_cfg("test"))), word_cfg("test"));
    })
}

#[test]
fn test_cfg_and() {
    with_default_globals(|| {
        let mut x = Cfg::False;
        x &= Cfg::True;
        assert_eq!(x, Cfg::False);

        x = word_cfg("test");
        x &= Cfg::False;
        assert_eq!(x, Cfg::False);

        x = word_cfg("test2");
        x &= Cfg::True;
        assert_eq!(x, word_cfg("test2"));

        x = Cfg::True;
        x &= word_cfg("test3");
        assert_eq!(x, word_cfg("test3"));

        x &= word_cfg("test4");
        assert_eq!(x, Cfg::All(vec![word_cfg("test3"), word_cfg("test4")]));

        x &= word_cfg("test5");
        assert_eq!(x, Cfg::All(vec![word_cfg("test3"), word_cfg("test4"), word_cfg("test5")]));

        x &= Cfg::All(vec![word_cfg("test6"), word_cfg("test7")]);
        assert_eq!(
            x,
            Cfg::All(vec![
                word_cfg("test3"),
                word_cfg("test4"),
                word_cfg("test5"),
                word_cfg("test6"),
                word_cfg("test7"),
            ])
        );

        let mut y = Cfg::Any(vec![word_cfg("a"), word_cfg("b")]);
        y &= x;
        assert_eq!(
            y,
            Cfg::All(vec![
                word_cfg("test3"),
                word_cfg("test4"),
                word_cfg("test5"),
                word_cfg("test6"),
                word_cfg("test7"),
                Cfg::Any(vec![word_cfg("a"), word_cfg("b")]),
            ])
        );

        assert_eq!(
            word_cfg("a") & word_cfg("b") & word_cfg("c"),
            Cfg::All(vec![word_cfg("a"), word_cfg("b"), word_cfg("c")])
        );
    })
}

#[test]
fn test_cfg_or() {
    with_default_globals(|| {
        let mut x = Cfg::True;
        x |= Cfg::False;
        assert_eq!(x, Cfg::True);

        x = word_cfg("test");
        x |= Cfg::True;
        assert_eq!(x, Cfg::True);

        x = word_cfg("test2");
        x |= Cfg::False;
        assert_eq!(x, word_cfg("test2"));

        x = Cfg::False;
        x |= word_cfg("test3");
        assert_eq!(x, word_cfg("test3"));

        x |= word_cfg("test4");
        assert_eq!(x, Cfg::Any(vec![word_cfg("test3"), word_cfg("test4")]));

        x |= word_cfg("test5");
        assert_eq!(x, Cfg::Any(vec![word_cfg("test3"), word_cfg("test4"), word_cfg("test5")]));

        x |= Cfg::Any(vec![word_cfg("test6"), word_cfg("test7")]);
        assert_eq!(
            x,
            Cfg::Any(vec![
                word_cfg("test3"),
                word_cfg("test4"),
                word_cfg("test5"),
                word_cfg("test6"),
                word_cfg("test7"),
            ])
        );

        let mut y = Cfg::All(vec![word_cfg("a"), word_cfg("b")]);
        y |= x;
        assert_eq!(
            y,
            Cfg::Any(vec![
                word_cfg("test3"),
                word_cfg("test4"),
                word_cfg("test5"),
                word_cfg("test6"),
                word_cfg("test7"),
                Cfg::All(vec![word_cfg("a"), word_cfg("b")]),
            ])
        );

        assert_eq!(
            word_cfg("a") | word_cfg("b") | word_cfg("c"),
            Cfg::Any(vec![word_cfg("a"), word_cfg("b"), word_cfg("c")])
        );
    })
}

#[test]
fn test_parse_ok() {
    with_default_globals(|| {
        let mi = dummy_meta_item_word("all");
        assert_eq!(Cfg::parse(&mi), Ok(word_cfg("all")));

        let mi =
            attr::mk_name_value_item_str(Ident::from_str("all"), Symbol::intern("done"), DUMMY_SP);
        assert_eq!(Cfg::parse(&mi), Ok(name_value_cfg("all", "done")));

        let mi = dummy_meta_item_list!(all, [a, b]);
        assert_eq!(Cfg::parse(&mi), Ok(word_cfg("a") & word_cfg("b")));

        let mi = dummy_meta_item_list!(any, [a, b]);
        assert_eq!(Cfg::parse(&mi), Ok(word_cfg("a") | word_cfg("b")));

        let mi = dummy_meta_item_list!(not, [a]);
        assert_eq!(Cfg::parse(&mi), Ok(!word_cfg("a")));

        let mi = dummy_meta_item_list!(
            not,
            [dummy_meta_item_list!(
                any,
                [dummy_meta_item_word("a"), dummy_meta_item_list!(all, [b, c]),]
            ),]
        );
        assert_eq!(Cfg::parse(&mi), Ok(!(word_cfg("a") | (word_cfg("b") & word_cfg("c")))));

        let mi = dummy_meta_item_list!(all, [a, b, c]);
        assert_eq!(Cfg::parse(&mi), Ok(word_cfg("a") & word_cfg("b") & word_cfg("c")));
    })
}

#[test]
fn test_parse_err() {
    with_default_globals(|| {
        let mi = attr::mk_name_value_item(Ident::from_str("foo"), LitKind::Bool(false), DUMMY_SP);
        assert!(Cfg::parse(&mi).is_err());

        let mi = dummy_meta_item_list!(not, [a, b]);
        assert!(Cfg::parse(&mi).is_err());

        let mi = dummy_meta_item_list!(not, []);
        assert!(Cfg::parse(&mi).is_err());

        let mi = dummy_meta_item_list!(foo, []);
        assert!(Cfg::parse(&mi).is_err());

        let mi = dummy_meta_item_list!(
            all,
            [dummy_meta_item_list!(foo, []), dummy_meta_item_word("b"),]
        );
        assert!(Cfg::parse(&mi).is_err());

        let mi = dummy_meta_item_list!(
            any,
            [dummy_meta_item_word("a"), dummy_meta_item_list!(foo, []),]
        );
        assert!(Cfg::parse(&mi).is_err());

        let mi = dummy_meta_item_list!(not, [dummy_meta_item_list!(foo, []),]);
        assert!(Cfg::parse(&mi).is_err());
    })
}

#[test]
fn test_render_short_html() {
    with_default_globals(|| {
        assert_eq!(word_cfg("unix").render_short_html(), "Unix");
        assert_eq!(name_value_cfg("target_os", "macos").render_short_html(), "macOS");
        assert_eq!(name_value_cfg("target_pointer_width", "16").render_short_html(), "16-bit");
        assert_eq!(name_value_cfg("target_endian", "little").render_short_html(), "Little-endian");
        assert_eq!((!word_cfg("windows")).render_short_html(), "Non-Windows");
        assert_eq!(
            (word_cfg("unix") & word_cfg("windows")).render_short_html(),
            "Unix and Windows"
        );
        assert_eq!((word_cfg("unix") | word_cfg("windows")).render_short_html(), "Unix or Windows");
        assert_eq!(
            (word_cfg("unix") & word_cfg("windows") & word_cfg("debug_assertions"))
                .render_short_html(),
            "Unix and Windows and debug-assertions enabled"
        );
        assert_eq!(
            (word_cfg("unix") | word_cfg("windows") | word_cfg("debug_assertions"))
                .render_short_html(),
            "Unix or Windows or debug-assertions enabled"
        );
        assert_eq!(
            (!(word_cfg("unix") | word_cfg("windows") | word_cfg("debug_assertions")))
                .render_short_html(),
            "Neither Unix nor Windows nor debug-assertions enabled"
        );
        assert_eq!(
            ((word_cfg("unix") & name_value_cfg("target_arch", "x86_64"))
                | (word_cfg("windows") & name_value_cfg("target_pointer_width", "64")))
            .render_short_html(),
            "Unix and x86-64, or Windows and 64-bit"
        );
        assert_eq!(
            (!(word_cfg("unix") & word_cfg("windows"))).render_short_html(),
            "Not (Unix and Windows)"
        );
        assert_eq!(
            ((word_cfg("debug_assertions") | word_cfg("windows")) & word_cfg("unix"))
                .render_short_html(),
            "(Debug-assertions enabled or Windows) and Unix"
        );
        assert_eq!(
            name_value_cfg("target_feature", "sse2").render_short_html(),
            "<code>sse2</code>"
        );
        assert_eq!(
            (name_value_cfg("target_arch", "x86_64") & name_value_cfg("target_feature", "sse2"))
                .render_short_html(),
            "x86-64 and <code>sse2</code>"
        );
    })
}

#[test]
fn test_render_long_html() {
    with_default_globals(|| {
        assert_eq!(
            word_cfg("unix").render_long_html(),
            "This is supported on <strong>Unix</strong> only."
        );
        assert_eq!(
            name_value_cfg("target_os", "macos").render_long_html(),
            "This is supported on <strong>macOS</strong> only."
        );
        assert_eq!(
            name_value_cfg("target_pointer_width", "16").render_long_html(),
            "This is supported on <strong>16-bit</strong> only."
        );
        assert_eq!(
            name_value_cfg("target_endian", "little").render_long_html(),
            "This is supported on <strong>little-endian</strong> only."
        );
        assert_eq!(
            (!word_cfg("windows")).render_long_html(),
            "This is supported on <strong>non-Windows</strong> only."
        );
        assert_eq!(
            (word_cfg("unix") & word_cfg("windows")).render_long_html(),
            "This is supported on <strong>Unix and Windows</strong> only."
        );
        assert_eq!(
            (word_cfg("unix") | word_cfg("windows")).render_long_html(),
            "This is supported on <strong>Unix or Windows</strong> only."
        );
        assert_eq!(
            (word_cfg("unix") & word_cfg("windows") & word_cfg("debug_assertions"))
                .render_long_html(),
            "This is supported on <strong>Unix and Windows and debug-assertions enabled\
                </strong> only."
        );
        assert_eq!(
            (word_cfg("unix") | word_cfg("windows") | word_cfg("debug_assertions"))
                .render_long_html(),
            "This is supported on <strong>Unix or Windows or debug-assertions enabled\
                </strong> only."
        );
        assert_eq!(
            (!(word_cfg("unix") | word_cfg("windows") | word_cfg("debug_assertions")))
                .render_long_html(),
            "This is supported on <strong>neither Unix nor Windows nor debug-assertions \
                enabled</strong>."
        );
        assert_eq!(
            ((word_cfg("unix") & name_value_cfg("target_arch", "x86_64"))
                | (word_cfg("windows") & name_value_cfg("target_pointer_width", "64")))
            .render_long_html(),
            "This is supported on <strong>Unix and x86-64, or Windows and 64-bit</strong> \
                only."
        );
        assert_eq!(
            (!(word_cfg("unix") & word_cfg("windows"))).render_long_html(),
            "This is supported on <strong>not (Unix and Windows)</strong>."
        );
        assert_eq!(
            ((word_cfg("debug_assertions") | word_cfg("windows")) & word_cfg("unix"))
                .render_long_html(),
            "This is supported on <strong>(debug-assertions enabled or Windows) and Unix\
            </strong> only."
        );
        assert_eq!(
            name_value_cfg("target_feature", "sse2").render_long_html(),
            "This is supported with <strong>target feature <code>sse2</code></strong> only."
        );
        assert_eq!(
            (name_value_cfg("target_arch", "x86_64") & name_value_cfg("target_feature", "sse2"))
                .render_long_html(),
            "This is supported on <strong>x86-64 and target feature \
            <code>sse2</code></strong> only."
        );
    })
}
