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
    mod foo {
        mod zed { }
    }
}
fn main(args: [str]/~) { baz(); }
