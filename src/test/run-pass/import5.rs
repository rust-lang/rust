use foo::bar;
mod foo {
    #[legacy_exports];
    use zed::bar;
    export bar;
    mod zed {
        #[legacy_exports];
        fn bar() { debug!("foo"); }
    }
}

fn main(args: ~[~str]) { bar(); }
