import foo::bar;
mod foo {
    import zed::bar;
    export bar;
    mod zed {
        fn bar() { #debug("foo"); }
    }
}

fn main(args: [str]) { bar(); }
