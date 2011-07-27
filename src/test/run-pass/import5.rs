import foo::bar;
mod foo {
    import zed::bar;
    export bar;
    mod zed {
        fn bar() { log "foo"; }
    }
}

fn main(args: vec[str]) { bar(); }