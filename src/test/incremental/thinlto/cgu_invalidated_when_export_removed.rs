// revisions: cfail1 cfail2
// build-pass

// rust-lang/rust#69798:
//
// This is analogous to cgu_invalidated_when_export_added, but it covers the
// other direction. This is analogous to cgu_invalidated_when_import_added: we
// include it, because it may uncover bugs in variant implementation strategies.

pub struct Foo {}
impl Drop for Foo {
    fn drop(&mut self) {
        println!("Dropping Foo");
    }
}
#[no_mangle]
pub extern "C" fn run() {
    thread_local! { pub static FOO : Foo = Foo { } ; }

    #[cfg(cfail1)]
    {
        FOO.with(|_f| ())
    }
}

pub fn main() { run() }
