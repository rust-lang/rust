pub trait Foo {
    // @has assoc_consts/trait.Foo.html '//pre[@class="rust item-decl"]' \
    //      'const FOO: usize = 13usize;'
    // @has - '//*[@id="associatedconstant.FOO"]' 'const FOO: usize'
    const FOO: usize = 12 + 1;
    // @has - '//*[@id="associatedconstant.FOO_NO_DEFAULT"]' 'const FOO_NO_DEFAULT: bool'
    const FOO_NO_DEFAULT: bool;
    // @!hasraw - FOO_HIDDEN
    #[doc(hidden)]
    const FOO_HIDDEN: u8 = 0;
}

pub struct Bar;

impl Foo for Bar {
    // @has assoc_consts/struct.Bar.html '//h3[@class="code-header"]' 'impl Foo for Bar'
    // @has - '//*[@id="associatedconstant.FOO"]' 'const FOO: usize'
    const FOO: usize = 12;
    // @has - '//*[@id="associatedconstant.FOO_NO_DEFAULT"]' 'const FOO_NO_DEFAULT: bool'
    const FOO_NO_DEFAULT: bool = false;
    // @!hasraw - FOO_HIDDEN
    #[doc(hidden)]
    const FOO_HIDDEN: u8 = 0;
}

impl Bar {
    // @has assoc_consts/struct.Bar.html '//*[@id="associatedconstant.BAR"]' \
    //      'const BAR: usize'
    pub const BAR: usize = 3;

    // @has - '//*[@id="associatedconstant.BAR_ESCAPED"]' \
    //      "const BAR_ESCAPED: &'static str = \"<em>markup</em>\""
    pub const BAR_ESCAPED: &'static str = "<em>markup</em>";
}

pub struct Baz<'a, U: 'a, T>(T, &'a [U]);

impl Bar {
    // @has assoc_consts/struct.Bar.html '//*[@id="associatedconstant.BAZ"]' \
    //      "const BAZ: Baz<'static, u8, u32>"
    pub const BAZ: Baz<'static, u8, u32> = Baz(321, &[1, 2, 3]);
}

pub fn f(_: &(ToString + 'static)) {}

impl Bar {
    // @has assoc_consts/struct.Bar.html '//*[@id="associatedconstant.F"]' \
    //      "const F: fn(_: &(dyn ToString + 'static))"
    pub const F: fn(_: &(ToString + 'static)) = f;
}

impl Bar {
    // @!hasraw assoc_consts/struct.Bar.html 'BAR_PRIVATE'
    const BAR_PRIVATE: char = 'a';
    // @!hasraw assoc_consts/struct.Bar.html 'BAR_HIDDEN'
    #[doc(hidden)]
    pub const BAR_HIDDEN: &'static str = "a";
}

// @has assoc_consts/trait.Qux.html
pub trait Qux {
    // @has - '//*[@id="associatedconstant.QUX0"]' 'const QUX0: u8'
    // @has - '//*[@class="docblock"]' "Docs for QUX0 in trait."
    /// Docs for QUX0 in trait.
    const QUX0: u8;
    // @has - '//*[@id="associatedconstant.QUX1"]' 'const QUX1: i8'
    // @has - '//*[@class="docblock"]' "Docs for QUX1 in trait."
    /// Docs for QUX1 in trait.
    const QUX1: i8;
    // @has - '//*[@id="associatedconstant.QUX_DEFAULT0"]' 'const QUX_DEFAULT0: u16'
    // @has - '//*[@class="docblock"]' "Docs for QUX_DEFAULT12 in trait."
    /// Docs for QUX_DEFAULT12 in trait.
    const QUX_DEFAULT0: u16 = 1;
    // @has - '//*[@id="associatedconstant.QUX_DEFAULT1"]' 'const QUX_DEFAULT1: i16'
    // @has - '//*[@class="docblock"]' "Docs for QUX_DEFAULT1 in trait."
    /// Docs for QUX_DEFAULT1 in trait.
    const QUX_DEFAULT1: i16 = 2;
    // @has - '//*[@id="associatedconstant.QUX_DEFAULT2"]' 'const QUX_DEFAULT2: u32'
    // @has - '//*[@class="docblock"]' "Docs for QUX_DEFAULT2 in trait."
    /// Docs for QUX_DEFAULT2 in trait.
    const QUX_DEFAULT2: u32 = 3;
}

// @has assoc_consts/struct.Bar.html '//h3[@class="code-header"]' 'impl Qux for Bar'
impl Qux for Bar {
    // @has - '//*[@id="associatedconstant.QUX0"]' 'const QUX0: u8'
    // @has - '//*[@class="docblock"]' "Docs for QUX0 in trait."
    /// Docs for QUX0 in trait.
    const QUX0: u8 = 4;
    // @has - '//*[@id="associatedconstant.QUX1"]' 'const QUX1: i8'
    // @has - '//*[@class="docblock"]' "Docs for QUX1 in impl."
    /// Docs for QUX1 in impl.
    const QUX1: i8 = 5;
    // @has - '//*[@id="associatedconstant.QUX_DEFAULT0"]' 'const QUX_DEFAULT0: u16'
    // @has - '//div[@class="impl-items"]//*[@class="docblock"]' "Docs for QUX_DEFAULT12 in trait."
    const QUX_DEFAULT0: u16 = 6;
    // @has - '//*[@id="associatedconstant.QUX_DEFAULT1"]' 'const QUX_DEFAULT1: i16'
    // @has - '//*[@class="docblock"]' "Docs for QUX_DEFAULT1 in impl."
    /// Docs for QUX_DEFAULT1 in impl.
    const QUX_DEFAULT1: i16 = 7;
    // @has - '//*[@id="associatedconstant.QUX_DEFAULT2"]' 'const QUX_DEFAULT2: u32'
    // @has - '//div[@class="impl-items"]//*[@class="docblock"]' "Docs for QUX_DEFAULT2 in trait."
}
