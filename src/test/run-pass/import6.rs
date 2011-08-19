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
}
fn main() { baz(); }
