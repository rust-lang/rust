use rustc_ast::ast::LitIntType;
use rustc_ast::{MetaItemInner, MetaItemLit, Path, Safety, StrStyle};
use rustc_span::symbol::{Ident, kw};
use rustc_span::{DUMMY_SP, create_default_session_globals_then};
use thin_vec::thin_vec;

use super::*;

fn word_cfg(s: &str) -> Cfg {
    Cfg::Cfg(Symbol::intern(s), None)
}

fn name_value_cfg(name: &str, value: &str) -> Cfg {
    Cfg::Cfg(Symbol::intern(name), Some(Symbol::intern(value)))
}

fn dummy_lit(symbol: Symbol, kind: LitKind) -> MetaItemInner {
    MetaItemInner::Lit(MetaItemLit { symbol, suffix: None, kind, span: DUMMY_SP })
}

fn dummy_meta_item_word(name: &str) -> MetaItemInner {
    MetaItemInner::MetaItem(MetaItem {
        unsafety: Safety::Default,
        path: Path::from_ident(Ident::from_str(name)),
        kind: MetaItemKind::Word,
        span: DUMMY_SP,
    })
}

fn dummy_meta_item_name_value(name: &str, symbol: Symbol, kind: LitKind) -> MetaItemInner {
    let lit = MetaItemLit { symbol, suffix: None, kind, span: DUMMY_SP };
    MetaItemInner::MetaItem(MetaItem {
        unsafety: Safety::Default,
        path: Path::from_ident(Ident::from_str(name)),
        kind: MetaItemKind::NameValue(lit),
        span: DUMMY_SP,
    })
}

macro_rules! dummy_meta_item_list {
    ($name:ident, [$($list:ident),* $(,)?]) => {
        MetaItemInner::MetaItem(MetaItem {
            unsafety: Safety::Default,
            path: Path::from_ident(Ident::from_str(stringify!($name))),
            kind: MetaItemKind::List(thin_vec![
                $(
                    dummy_meta_item_word(stringify!($list)),
                )*
            ]),
            span: DUMMY_SP,
        })
    };

    ($name:ident, [$($list:expr),* $(,)?]) => {
        MetaItemInner::MetaItem(MetaItem {
            unsafety: Safety::Default,
            path: Path::from_ident(Ident::from_str(stringify!($name))),
            kind: MetaItemKind::List(thin_vec![
                $($list,)*
            ]),
            span: DUMMY_SP,
        })
    };
}

#[test]
fn test_cfg_not() {
    create_default_session_globals_then(|| {
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
    create_default_session_globals_then(|| {
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

        x &= word_cfg("test3");
        assert_eq!(x, word_cfg("test3"));

        x &= word_cfg("test4");
        assert_eq!(x, Cfg::All(vec![word_cfg("test3"), word_cfg("test4")]));

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

        let mut z = word_cfg("test8");
        z &= Cfg::All(vec![word_cfg("test9"), word_cfg("test10")]);
        assert_eq!(z, Cfg::All(vec![word_cfg("test9"), word_cfg("test10"), word_cfg("test8")]));

        let mut z = word_cfg("test11");
        z &= Cfg::All(vec![word_cfg("test11"), word_cfg("test12")]);
        assert_eq!(z, Cfg::All(vec![word_cfg("test11"), word_cfg("test12")]));

        assert_eq!(
            word_cfg("a") & word_cfg("b") & word_cfg("c"),
            Cfg::All(vec![word_cfg("a"), word_cfg("b"), word_cfg("c")])
        );
    })
}

#[test]
fn test_cfg_or() {
    create_default_session_globals_then(|| {
        let mut x = Cfg::True;
        x |= Cfg::False;
        assert_eq!(x, Cfg::True);

        x = word_cfg("test");
        x |= Cfg::True;
        assert_eq!(x, word_cfg("test"));

        x = word_cfg("test2");
        x |= Cfg::False;
        assert_eq!(x, word_cfg("test2"));

        x = Cfg::False;
        x |= word_cfg("test3");
        assert_eq!(x, word_cfg("test3"));

        x |= word_cfg("test3");
        assert_eq!(x, word_cfg("test3"));

        x |= word_cfg("test4");
        assert_eq!(x, Cfg::Any(vec![word_cfg("test3"), word_cfg("test4")]));

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

        let mut z = word_cfg("test8");
        z |= Cfg::Any(vec![word_cfg("test9"), word_cfg("test10")]);
        assert_eq!(z, Cfg::Any(vec![word_cfg("test9"), word_cfg("test10"), word_cfg("test8")]));

        let mut z = word_cfg("test11");
        z |= Cfg::Any(vec![word_cfg("test11"), word_cfg("test12")]);
        assert_eq!(z, Cfg::Any(vec![word_cfg("test11"), word_cfg("test12")]));

        assert_eq!(
            word_cfg("a") | word_cfg("b") | word_cfg("c"),
            Cfg::Any(vec![word_cfg("a"), word_cfg("b"), word_cfg("c")])
        );
    })
}

#[test]
fn test_parse_ok() {
    create_default_session_globals_then(|| {
        let r#true = Symbol::intern("true");
        let mi = dummy_lit(r#true, LitKind::Bool(true));
        assert_eq!(Cfg::parse(&mi), Ok(Cfg::True));

        let r#false = Symbol::intern("false");
        let mi = dummy_lit(r#false, LitKind::Bool(false));
        assert_eq!(Cfg::parse(&mi), Ok(Cfg::False));

        let mi = dummy_meta_item_word("all");
        assert_eq!(Cfg::parse(&mi), Ok(word_cfg("all")));

        let done = Symbol::intern("done");
        let mi = dummy_meta_item_name_value("all", done, LitKind::Str(done, StrStyle::Cooked));
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
    create_default_session_globals_then(|| {
        let mi = dummy_meta_item_name_value("foo", kw::False, LitKind::Bool(false));
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

        let c = Symbol::intern("e");
        let mi = dummy_lit(c, LitKind::Char('e'));
        assert!(Cfg::parse(&mi).is_err());

        let five = Symbol::intern("5");
        let mi = dummy_lit(five, LitKind::Int(5.into(), LitIntType::Unsuffixed));
        assert!(Cfg::parse(&mi).is_err());
    })
}

#[test]
fn test_render_short_html() {
    create_default_session_globals_then(|| {
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
    create_default_session_globals_then(|| {
        assert_eq!(word_cfg("unix").render_long_html(), "Available on <strong>Unix</strong> only.");
        assert_eq!(
            name_value_cfg("target_os", "macos").render_long_html(),
            "Available on <strong>macOS</strong> only."
        );
        assert_eq!(
            name_value_cfg("target_os", "wasi").render_long_html(),
            "Available on <strong>WASI</strong> only."
        );
        assert_eq!(
            name_value_cfg("target_pointer_width", "16").render_long_html(),
            "Available on <strong>16-bit</strong> only."
        );
        assert_eq!(
            name_value_cfg("target_endian", "little").render_long_html(),
            "Available on <strong>little-endian</strong> only."
        );
        assert_eq!(
            (!word_cfg("windows")).render_long_html(),
            "Available on <strong>non-Windows</strong> only."
        );
        assert_eq!(
            (word_cfg("unix") & word_cfg("windows")).render_long_html(),
            "Available on <strong>Unix and Windows</strong> only."
        );
        assert_eq!(
            (word_cfg("unix") | word_cfg("windows")).render_long_html(),
            "Available on <strong>Unix or Windows</strong> only."
        );
        assert_eq!(
            (word_cfg("unix") & word_cfg("windows") & word_cfg("debug_assertions"))
                .render_long_html(),
            "Available on <strong>Unix and Windows and debug-assertions enabled</strong> only."
        );
        assert_eq!(
            (word_cfg("unix") | word_cfg("windows") | word_cfg("debug_assertions"))
                .render_long_html(),
            "Available on <strong>Unix or Windows or debug-assertions enabled</strong> only."
        );
        assert_eq!(
            (!(word_cfg("unix") | word_cfg("windows") | word_cfg("debug_assertions")))
                .render_long_html(),
            "Available on <strong>neither Unix nor Windows nor debug-assertions enabled</strong>."
        );
        assert_eq!(
            ((word_cfg("unix") & name_value_cfg("target_arch", "x86_64"))
                | (word_cfg("windows") & name_value_cfg("target_pointer_width", "64")))
            .render_long_html(),
            "Available on <strong>Unix and x86-64, or Windows and 64-bit</strong> only."
        );
        assert_eq!(
            (!(word_cfg("unix") & word_cfg("windows"))).render_long_html(),
            "Available on <strong>not (Unix and Windows)</strong>."
        );
        assert_eq!(
            ((word_cfg("debug_assertions") | word_cfg("windows")) & word_cfg("unix"))
                .render_long_html(),
            "Available on <strong>(debug-assertions enabled or Windows) and Unix</strong> only."
        );
        assert_eq!(
            name_value_cfg("target_feature", "sse2").render_long_html(),
            "Available with <strong>target feature <code>sse2</code></strong> only."
        );
        assert_eq!(
            (name_value_cfg("target_arch", "x86_64") & name_value_cfg("target_feature", "sse2"))
                .render_long_html(),
            "Available on <strong>x86-64 and target feature <code>sse2</code></strong> only."
        );
    })
}

#[test]
fn test_simplify_with() {
    // This is a tiny subset of things that could be simplified, but it likely covers 90% of
    // real world usecases well.
    create_default_session_globals_then(|| {
        let foo = word_cfg("foo");
        let bar = word_cfg("bar");
        let baz = word_cfg("baz");
        let quux = word_cfg("quux");

        let foobar = Cfg::All(vec![foo.clone(), bar.clone()]);
        let barbaz = Cfg::All(vec![bar.clone(), baz.clone()]);
        let foobarbaz = Cfg::All(vec![foo.clone(), bar.clone(), baz.clone()]);
        let bazquux = Cfg::All(vec![baz.clone(), quux.clone()]);

        // Unrelated cfgs don't affect each other
        assert_eq!(foo.simplify_with(&bar).as_ref(), Some(&foo));
        assert_eq!(foobar.simplify_with(&bazquux).as_ref(), Some(&foobar));

        // Identical cfgs are eliminated
        assert_eq!(foo.simplify_with(&foo), None);
        assert_eq!(foobar.simplify_with(&foobar), None);

        // Multiple cfgs eliminate a single assumed cfg
        assert_eq!(foobar.simplify_with(&foo).as_ref(), Some(&bar));
        assert_eq!(foobar.simplify_with(&bar).as_ref(), Some(&foo));

        // A single cfg is eliminated by multiple assumed cfg containing it
        assert_eq!(foo.simplify_with(&foobar), None);

        // Multiple cfgs eliminate the matching subset of multiple assumed cfg
        assert_eq!(foobar.simplify_with(&barbaz).as_ref(), Some(&foo));
        assert_eq!(foobar.simplify_with(&foobarbaz), None);
    });
}
