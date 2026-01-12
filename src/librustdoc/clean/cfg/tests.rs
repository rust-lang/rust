use rustc_data_structures::thin_vec::thin_vec;
use rustc_hir::attrs::CfgEntry;
use rustc_span::{DUMMY_SP, create_default_session_globals_then};

use super::*;

fn word_cfg(name: &str) -> Cfg {
    Cfg(word_cfg_e(name))
}

fn word_cfg_e(name: &str) -> CfgEntry {
    CfgEntry::NameValue { name: Symbol::intern(name), value: None, span: DUMMY_SP }
}

fn name_value_cfg(name: &str, value: &str) -> Cfg {
    Cfg(name_value_cfg_e(name, value))
}

fn name_value_cfg_e(name: &str, value: &str) -> CfgEntry {
    CfgEntry::NameValue {
        name: Symbol::intern(name),

        value: Some(Symbol::intern(value)),
        span: DUMMY_SP,
    }
}

fn cfg_all(v: ThinVec<CfgEntry>) -> Cfg {
    Cfg(cfg_all_e(v))
}

fn cfg_all_e(v: ThinVec<CfgEntry>) -> CfgEntry {
    CfgEntry::All(v, DUMMY_SP)
}

fn cfg_any(v: ThinVec<CfgEntry>) -> Cfg {
    Cfg(cfg_any_e(v))
}

fn cfg_any_e(v: ThinVec<CfgEntry>) -> CfgEntry {
    CfgEntry::Any(v, DUMMY_SP)
}

fn cfg_not(v: CfgEntry) -> Cfg {
    Cfg(CfgEntry::Not(Box::new(v), DUMMY_SP))
}

fn cfg_true() -> Cfg {
    Cfg(CfgEntry::Bool(true, DUMMY_SP))
}

fn cfg_false() -> Cfg {
    Cfg(CfgEntry::Bool(false, DUMMY_SP))
}

#[test]
fn test_cfg_not() {
    create_default_session_globals_then(|| {
        assert_eq!(!cfg_false(), cfg_true());
        assert_eq!(!cfg_true(), cfg_false());
        assert_eq!(!word_cfg("test"), cfg_not(word_cfg_e("test")));
        assert_eq!(
            !cfg_all(thin_vec![word_cfg_e("a"), word_cfg_e("b")]),
            cfg_not(cfg_all_e(thin_vec![word_cfg_e("a"), word_cfg_e("b")]))
        );
        assert_eq!(
            !cfg_any(thin_vec![word_cfg_e("a"), word_cfg_e("b")]),
            cfg_not(cfg_any_e(thin_vec![word_cfg_e("a"), word_cfg_e("b")]))
        );
        assert_eq!(!cfg_not(word_cfg_e("test")), word_cfg("test"));
    })
}

#[test]
fn test_cfg_and() {
    create_default_session_globals_then(|| {
        let mut x = cfg_false();
        x &= cfg_true();
        assert_eq!(x, cfg_false());

        x = word_cfg("test");
        x &= cfg_false();
        assert_eq!(x, cfg_false());

        x = word_cfg("test2");
        x &= cfg_true();
        assert_eq!(x, word_cfg("test2"));

        x = cfg_true();
        x &= word_cfg("test3");
        assert_eq!(x, word_cfg("test3"));

        x &= word_cfg("test3");
        assert_eq!(x, word_cfg("test3"));

        x &= word_cfg("test4");
        assert_eq!(x, cfg_all(thin_vec![word_cfg_e("test3"), word_cfg_e("test4")]));

        x &= word_cfg("test4");
        assert_eq!(x, cfg_all(thin_vec![word_cfg_e("test3"), word_cfg_e("test4")]));

        x &= word_cfg("test5");
        assert_eq!(
            x,
            cfg_all(thin_vec![word_cfg_e("test3"), word_cfg_e("test4"), word_cfg_e("test5")])
        );

        x &= cfg_all(thin_vec![word_cfg_e("test6"), word_cfg_e("test7")]);
        assert_eq!(
            x,
            cfg_all(thin_vec![
                word_cfg_e("test3"),
                word_cfg_e("test4"),
                word_cfg_e("test5"),
                word_cfg_e("test6"),
                word_cfg_e("test7"),
            ])
        );

        x &= cfg_all(thin_vec![word_cfg_e("test6"), word_cfg_e("test7")]);
        assert_eq!(
            x,
            cfg_all(thin_vec![
                word_cfg_e("test3"),
                word_cfg_e("test4"),
                word_cfg_e("test5"),
                word_cfg_e("test6"),
                word_cfg_e("test7"),
            ])
        );

        let mut y = cfg_any(thin_vec![word_cfg_e("a"), word_cfg_e("b")]);
        y &= x;
        assert_eq!(
            y,
            cfg_all(thin_vec![
                word_cfg_e("test3"),
                word_cfg_e("test4"),
                word_cfg_e("test5"),
                word_cfg_e("test6"),
                word_cfg_e("test7"),
                cfg_any_e(thin_vec![word_cfg_e("a"), word_cfg_e("b")]),
            ])
        );

        let mut z = word_cfg("test8");
        z &= cfg_all(thin_vec![word_cfg_e("test9"), word_cfg_e("test10")]);
        assert_eq!(
            z,
            cfg_all(thin_vec![word_cfg_e("test9"), word_cfg_e("test10"), word_cfg_e("test8"),]),
        );

        let mut z = word_cfg("test11");
        z &= cfg_all(thin_vec![word_cfg_e("test11"), word_cfg_e("test12")]);
        assert_eq!(z, cfg_all(thin_vec![word_cfg_e("test11"), word_cfg_e("test12")]));

        assert_eq!(
            word_cfg("a") & word_cfg("b") & word_cfg("c"),
            cfg_all(thin_vec![word_cfg_e("a"), word_cfg_e("b"), word_cfg_e("c")])
        );
    })
}

#[test]
fn test_cfg_or() {
    create_default_session_globals_then(|| {
        let mut x = cfg_true();
        x |= cfg_false();
        assert_eq!(x, cfg_true());

        x = word_cfg("test");
        x |= cfg_true();
        assert_eq!(x, word_cfg("test"));

        x = word_cfg("test2");
        x |= cfg_false();
        assert_eq!(x, word_cfg("test2"));

        x = cfg_false();
        x |= word_cfg("test3");
        assert_eq!(x, word_cfg("test3"));

        x |= word_cfg("test3");
        assert_eq!(x, word_cfg("test3"));

        x |= word_cfg("test4");
        assert_eq!(x, cfg_any(thin_vec![word_cfg_e("test3"), word_cfg_e("test4")]));

        x |= word_cfg("test4");
        assert_eq!(x, cfg_any(thin_vec![word_cfg_e("test3"), word_cfg_e("test4")]));

        x |= word_cfg("test5");
        assert_eq!(
            x,
            cfg_any(thin_vec![word_cfg_e("test3"), word_cfg_e("test4"), word_cfg_e("test5")])
        );

        x |= cfg_any(thin_vec![word_cfg_e("test6"), word_cfg_e("test7")]);
        assert_eq!(
            x,
            cfg_any(thin_vec![
                word_cfg_e("test3"),
                word_cfg_e("test4"),
                word_cfg_e("test5"),
                word_cfg_e("test6"),
                word_cfg_e("test7"),
            ])
        );

        x |= cfg_any(thin_vec![word_cfg_e("test6"), word_cfg_e("test7")]);
        assert_eq!(
            x,
            cfg_any(thin_vec![
                word_cfg_e("test3"),
                word_cfg_e("test4"),
                word_cfg_e("test5"),
                word_cfg_e("test6"),
                word_cfg_e("test7"),
            ])
        );

        let mut y = cfg_all(thin_vec![word_cfg_e("a"), word_cfg_e("b")]);
        y |= x;
        assert_eq!(
            y,
            cfg_any(thin_vec![
                word_cfg_e("test3"),
                word_cfg_e("test4"),
                word_cfg_e("test5"),
                word_cfg_e("test6"),
                word_cfg_e("test7"),
                cfg_all_e(thin_vec![word_cfg_e("a"), word_cfg_e("b")]),
            ])
        );

        let mut z = word_cfg("test8");
        z |= cfg_any(thin_vec![word_cfg_e("test9"), word_cfg_e("test10")]);
        assert_eq!(
            z,
            cfg_any(thin_vec![word_cfg_e("test9"), word_cfg_e("test10"), word_cfg_e("test8")])
        );

        let mut z = word_cfg("test11");
        z |= cfg_any(thin_vec![word_cfg_e("test11"), word_cfg_e("test12")]);
        assert_eq!(z, cfg_any(thin_vec![word_cfg_e("test11"), word_cfg_e("test12")]));

        assert_eq!(
            word_cfg("a") | word_cfg("b") | word_cfg("c"),
            cfg_any(thin_vec![word_cfg_e("a"), word_cfg_e("b"), word_cfg_e("c")])
        );
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
        let foo = word_cfg_e("foo");
        let bar = word_cfg_e("bar");
        let baz = word_cfg_e("baz");
        let quux = word_cfg_e("quux");

        let foobar = cfg_all(thin_vec![foo.clone(), bar.clone()]);
        let barbaz = cfg_all(thin_vec![bar.clone(), baz.clone()]);
        let foobarbaz = cfg_all(thin_vec![foo.clone(), bar.clone(), baz.clone()]);
        let bazquux = cfg_all(thin_vec![baz.clone(), quux.clone()]);

        // Unrelated cfgs don't affect each other
        assert_eq!(
            Cfg(foo.clone()).simplify_with(&Cfg(bar.clone())).as_ref(),
            Some(&Cfg(foo.clone()))
        );
        assert_eq!(foobar.simplify_with(&bazquux).as_ref(), Some(&foobar));

        // Identical cfgs are eliminated
        assert_eq!(Cfg(foo.clone()).simplify_with(&Cfg(foo.clone())), None);
        assert_eq!(foobar.simplify_with(&foobar), None);

        // Multiple cfgs eliminate a single assumed cfg
        assert_eq!(foobar.simplify_with(&Cfg(foo.clone())).as_ref(), Some(&Cfg(bar.clone())));
        assert_eq!(foobar.simplify_with(&Cfg(bar)).as_ref(), Some(&Cfg(foo.clone())));

        // A single cfg is eliminated by multiple assumed cfg containing it
        assert_eq!(Cfg(foo.clone()).simplify_with(&foobar), None);

        // Multiple cfgs eliminate the matching subset of multiple assumed cfg
        assert_eq!(foobar.simplify_with(&barbaz).as_ref(), Some(&Cfg(foo)));
        assert_eq!(foobar.simplify_with(&foobarbaz), None);
    });
}
