use foo::bar;
mod foo {
    use zed::bar;
    export bar;
    mod zed {
        fn bar() { debug!("foo"); }
    }
}

fn main(args: ~[~str]) { bar(); }
