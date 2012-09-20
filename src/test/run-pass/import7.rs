use foo::zed;
use bar::baz;
mod foo {
    #[legacy_exports];
    mod zed {
        #[legacy_exports];
        fn baz() { debug!("baz"); }
    }
}
mod bar {
    #[legacy_exports];
    use zed::baz;
    export baz;
    mod foo {
        #[legacy_exports];
        mod zed {
            #[legacy_exports]; }
    }
}
fn main() { baz(); }
