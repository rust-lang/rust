//@ edition: 2024
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

//@ has foo/macro.macro_rule.html '//*[@class="code-attribute"]' '#[unsafe(link_section = ".text")]'
#[unsafe(link_section = ".text")]
#[macro_export]
macro_rules! macro_rule {
    () => {};
}

//@ has 'foo/enum.Enum.html' '//*[@class="code-attribute"]' '#[unsafe(link_section = "enum")]'
#[unsafe(link_section = "enum")]
pub enum Enum {
    //@ has 'foo/enum.Enum.html' '//*[@class="code-attribute"]' '#[unsafe(link_section = "a")]'
    //@ has - '//*[@class="variants"]//*[@class="code-attribute"]' '#[unsafe(link_section = "a")]'
    #[unsafe(link_section = "a")]
    A,
    //@ has 'foo/enum.Enum.html' '//*[@class="code-attribute"]' '#[unsafe(link_section = "quz")]'
    //@ has - '//*[@class="variants"]//*[@class="code-attribute"]' '#[unsafe(link_section = "quz")]'
    #[unsafe(link_section = "quz")]
    Quz {
        //@ has 'foo/enum.Enum.html' '//*[@class="code-attribute"]' '#[unsafe(link_section = "b")]'
        //@ has - '//*[@class="variants"]//*[@class="code-attribute"]' '#[unsafe(link_section = "b")]'
        #[unsafe(link_section = "b")]
        b: (),
    },
}

//@ has 'foo/trait.Trait.html' '//*[@class="code-attribute"]' '#[unsafe(link_section = "trait")]'
#[unsafe(link_section = "trait")]
pub trait Trait {
    //@ has 'foo/trait.Trait.html'
    //@ has - '//*[@id="associatedconstant.BAR"]/*[@class="code-header"]/*[@class="code-attribute"]' '#[unsafe(link_section = "bar")]'
    //@ has - '//*[@id="associatedconstant.BAR"]/*[@class="code-header"]' 'const BAR: u32 = 0u32'
    #[unsafe(link_section = "bar")]
    const BAR: u32 = 0;

    //@ has - '//*[@class="code-attribute"]' '#[unsafe(link_section = "foo")]'
    #[unsafe(link_section = "foo")]
    fn foo() {}
}

//@ has 'foo/union.Union.html' '//*[@class="code-attribute"]' '#[unsafe(link_section = "union")]'
#[unsafe(link_section = "union")]
pub union Union {
    //@ has 'foo/union.Union.html' '//*[@class="code-attribute"]' '#[unsafe(link_section = "x")]'
    #[unsafe(link_section = "x")]
    pub x: u32,
    y: f32,
}

//@ has 'foo/struct.Struct.html' '//*[@class="code-attribute"]' '#[unsafe(link_section = "struct")]'
#[unsafe(link_section = "struct")]
pub struct Struct {
    //@ has 'foo/struct.Struct.html' '//*[@class="code-attribute"]' '#[unsafe(link_section = "x")]'
    //@ has - '//*[@id="structfield.x"]//*[@class="code-attribute"]' '#[unsafe(link_section = "x")]'
    #[unsafe(link_section = "x")]
    pub x: u32,
    y: f32,
}

// Check that the attributes from the trait items show up consistently in the impl.
//@ has 'foo/struct.Struct.html' '//*[@id="trait-implementations-list"]//*[@class="code-attribute"]' '#[unsafe(link_section = "bar")]'
//@ has 'foo/struct.Struct.html' '//*[@id="trait-implementations-list"]//*[@class="code-attribute"]' '#[unsafe(link_section = "foo")]'
impl Trait for Struct {}
