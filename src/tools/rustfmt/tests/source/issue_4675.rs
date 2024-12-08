macro_rules! foo {
    ($s:ident ( $p:pat )) => {
        Foo {
            name: Name::$s($p),
            ..
    }
    };
}