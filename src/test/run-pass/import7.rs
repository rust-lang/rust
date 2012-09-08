use foo::zed;
use bar::baz;
mod foo {
    mod zed {
        fn baz() { debug!("baz"); }
    }
}
mod bar {
    use zed::baz;
    export baz;
    mod foo {
        mod zed { }
    }
}
fn main(args: ~[~str]) { baz(); }
