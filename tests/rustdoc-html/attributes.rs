//@ edition: 2024
//@ only-linux
#![crate_name = "foo"]

//@ has foo/fn.f.html '//*[@class="code-attribute"]' '#[unsafe(no_mangle)]'
#[unsafe(no_mangle)]
pub extern "C" fn f() {}

//@ has foo/fn.g.html '//*[@class="code-attribute"]' '#[unsafe(export_name = "bar")]'
#[unsafe(export_name = "bar")]
pub extern "C" fn g() {}

//@ has foo/fn.escape_special.html '//*[@class="code-attribute"]' \
//                                 '#[unsafe(export_name = "\n\"\n")]'
#[unsafe(export_name = "\n\"
")]
pub extern "C" fn escape_special() {}

// issue: <https://github.com/rust-lang/rust/issues/142835>
//@ has foo/fn.escape_html.html '//*[@class="code-attribute"]' \
//                              '#[unsafe(export_name = "<script>alert()</script>")]'
#[unsafe(export_name = "<script>alert()</script>")]
pub extern "C" fn escape_html() {}

//@ has foo/fn.example.html '//*[@class="code-attribute"]' '#[unsafe(link_section = ".text")]'
#[unsafe(link_section = ".text")]
pub extern "C" fn example() {}

//@ has foo/struct.Repr.html '//*[@class="code-attribute"]' '#[repr(C, align(8))]'
#[repr(C, align(8))]
pub struct Repr;

pub trait Trait {
    //@ has 'foo/trait.Trait.html' '//*[@class="code-attribute"]' '#[unsafe(link_section = "extra")]'
    #[unsafe(link_section = "extra")]
    fn func() {}
}

pub struct Struct;

// Check that the attributes from the trait items show up consistently in the impl.
//@ has 'foo/struct.Struct.html' '//*[@id="trait-implementations-list"]//*[@class="code-attribute"]' '#[unsafe(link_section = "extra")]'
impl Trait for Struct {}
