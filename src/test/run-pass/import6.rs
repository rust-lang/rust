import foo::zed;
import bar::baz;
mod foo {
    mod zed {
        fn baz() { #debug("baz"); }
    }
}
mod bar {
    import zed::baz;
    export baz;
}
fn main() { baz(); }
