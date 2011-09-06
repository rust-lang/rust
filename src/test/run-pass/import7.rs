import bar::baz;
import foo::zed;
mod foo {
    mod zed {
        fn baz() { log "baz"; }
    }
}
mod bar {
    import zed::baz;
    export baz;
    mod foo {
        mod zed { }
    }
}
fn main(args: [str]) { baz(); }
