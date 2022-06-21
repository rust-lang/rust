// Testing the formatting of constant values (i.e. evaluated constant expressions)
// where the specific format was first proposed in issue #98929.

// edition:2021
#![crate_name = "consts"]

// aux-build:const-value.rs
// build-aux-docs
// ignore-cross-compile
extern crate const_value as aux;

// ignore-tidy-linelength

// FIXME: Some tests in here might be redundant and already present in other test files.
// FIXME: Test restricted visibilities (e.g. `pub(super)`, `pub(in crate::some::thing)`).

// Check that constant expressions are printed in their evaluated form.
//
// @has 'consts/constant.HOUR_IN_SECONDS.html'
// @has - '//*[@class="docblock item-decl"]//code' 'pub const HOUR_IN_SECONDS: u64 = 3600;'
pub const HOUR_IN_SECONDS: u64 = 60 * 60;

// @has 'consts/constant.NEGATIVE.html'
// @has - '//*[@class="docblock item-decl"]//code' 'pub const NEGATIVE: i64 = -3600;'
pub const NEGATIVE: i64 = -60 * 60;

// @has 'consts/constant.CONCATENATED.html'
// @has - '//*[@class="docblock item-decl"]//code' \
//        "pub const CONCATENATED: &'static str = \"[0, +∞)\";"
pub const CONCATENATED: &str = concat!("[", stringify!(0), ", ", "+∞", ")");

// @has 'consts/struct.Record.html'
pub struct Record<'r> {
    pub one: &'r str,
    pub two: (i32,),
}

// Test that structs whose fields are all public and 1-tuples are displayed correctly.
// Furthermore, the struct fields should appear in definition order.
// Re snapshot: Check that hyperlinks are generated for the struct name and fields.
//
// @has 'consts/constant.REC.html'
// @has - '//*[@class="docblock item-decl"]//code' \
//        "const REC: Record<'static> = Record { one: \"thriving\", two: (180,) }"
// @snapshot rec - '//*[@class="docblock item-decl"]//code'
pub const REC: Record<'_> = {
    assert!(true);

    let auxiliary = 90 * "||".len() as i32;
    Record {
        two: (
            auxiliary,
            #[cfg(FALSE)]
            "vanished",
        ),
        one: "thriving",
    }
};

// Check that private and doc(hidden) struct fields are not displayed.
// Instead, an ellipsis (namely `..`) should be printed.
// Re snapshot: Check that hyperlinks are generated for the struct name and the public struct field.
//
// @has 'consts/constant.STRUCT.html'
// @has - '//*[@class="docblock item-decl"]//code' \
//        'const STRUCT: Struct = Struct { public: (), .. }'
// @snapshot struct - '//*[@class="docblock item-decl"]//code'
pub const STRUCT: Struct = Struct {
    private : /* SourceMap::span_to_snippet trap */ (),
    public: { 1 + 3; },
    hidden: ()
};

// Test that enum variants, 2-tuples, bools and structs (with private and doc(hidden) fields) nested
// within are rendered correctly. Further, check that there is a maximum depth.
// Re snapshot: Test the hyperlinks are generated for the cross-crate enum variant etc.
//
// @has 'consts/constant.NESTED.html'
// @has - '//*[@class="docblock item-decl"]//code' \
//        'const NESTED: Option<(Struct, bool)> = Some((Struct { public: …, .. }, false))'
// @snapshot nested - '//*[@class="docblock item-decl"]//code'
pub const NESTED: Option<(Struct, bool)> = Some((
    Struct {
        public: (),
        private: (),
        hidden: (),
    },
    false,
));

use std::sync::atomic::AtomicBool;

// @has 'consts/struct.Struct.html'
pub struct Struct {
    private: (),
    pub public: (),
    #[doc(hidden)]
    pub hidden: (),
}

impl Struct {
    // Check that even inside inherent impl blocks private and doc(hidden) struct fields
    // are not displayed.
    // Re snapshot: Check that hyperlinks are generated for the struct name and
    // the public struct field.
    //
    // @has - '//*[@id="associatedconstant.SELF"]' \
    //        'const SELF: Self = Struct { public: (), .. }'
    // @snapshot self - '//*[@id="associatedconstant.SELF"]//*[@class="code-header"]'
    pub const SELF: Self = Self {
        private: (),
        public: match () {
            () => {}
        },
        hidden: (),
    };

    // Verify that private and doc(hidden) *tuple* struct fields are not shown.
    // In their place, an underscore should be rendered.
    // Re snapshot: Check that a hyperlink is generated for the tuple struct name.
    //
    // @has - '//*[@id="associatedconstant.TUP_STRUCT"]' \
    //        'const TUP_STRUCT: TupStruct = TupStruct(_, -45, _, _)'
    // @snapshot tup-struct - '//*[@id="associatedconstant.TUP_STRUCT"]//*[@class="code-header"]'
    pub const TUP_STRUCT: TupStruct = TupStruct((), -45, (), false);

    // Check that structs whose fields are all doc(hidden) are rendered correctly.
    //
    // @has - '//*[@id="associatedconstant.SEALED0"]' \
    //        'const SEALED0: Container0 = Container0 { .. }'
    pub const SEALED0: Container0 = Container0 { hack: () };

    // Check that *tuple* structs whose fields are all private are rendered correctly.
    //
    // @has - '//*[@id="associatedconstant.SEALED1"]' \
    //        'const SEALED1: Container1 = Container1(_)'
    pub const SEALED1: Container1 = Container1(None);

    // Verify that cross-crate structs are displayed correctly and that their fields
    // are not leaked.
    // Re snapshot: Check that a hyperlink is generated for the name of the cross-crate struct.
    //
    // @has - '//*[@id="associatedconstant.SEALED2"]' \
    //        'const SEALED2: AtomicBool = AtomicBool { .. }'
    // @snapshot sealed2 - '//*[@id="associatedconstant.SEALED2"]//*[@class="code-header"]'
    pub const SEALED2: AtomicBool = AtomicBool::new(true);

    // Test that (local) *unit* enum variants are rendered properly.
    // Re snapshot: Test that a hyperlink is generated for the variant.
    //
    // @has - '//*[@id="associatedconstant.SUM0"]' \
    //        'const SUM0: Size = Uninhabited'
    // @snapshot sum0 - '//*[@id="associatedconstant.SUM0"]//*[@class="code-header"]'
    pub const SUM0: Size = self::Size::Uninhabited;

    // Test that (local) *struct* enum variants are rendered properly.
    // Re snapshot: Test that a hyperlink is generated for the variant.
    //
    // @has - '//*[@id="associatedconstant.SUM1"]' \
    //        'const SUM1: Size = Inhabited { inhabitants: 9000 }'
    // @snapshot sum1 - '//*[@id="associatedconstant.SUM1"]//*[@class="code-header"]'
    pub const SUM1: Size = AdtSize::Inhabited { inhabitants: 9_000 };

    // Test that (local) *tuple* enum variants are rendered properly.
    // Re snapshot: Test that a hyperlink is generated for the variant.
    //
    // @has - '//*[@id="associatedconstant.SUM2"]' \
    //        'const SUM2: Size = Unknown(Reason)'
    // @snapshot sum2 - '//*[@id="associatedconstant.SUM2"]//*[@class="code-header"]'
    pub const SUM2: Size = Size::Unknown(Reason);

    // @has - '//*[@id="associatedconstant.INT"]' \
    //        'const INT: i64 = 2368'
    pub const INT: i64 = 2345 + 23;

    // @has - '//*[@id="associatedconstant.STR"]' \
    //        "const STR: &'static str = \"hello friends\""
    pub const STR: &'static str = "hello friends";

    // @has - '//*[@id="associatedconstant.FLOAT0"]' \
    //        'const FLOAT0: f32 = 2930.21997'
    pub const FLOAT0: f32 = 2930.21997;

    // @has - '//*[@id="associatedconstant.FLOAT1"]' \
    //        'const FLOAT1: f64 = -3.42E+21'
    pub const FLOAT1: f64 = -3.42e+21;

    // @has - '//*[@id="associatedconstant.REF"]' \
    //        "const REF: &'static i32 = _"
    pub const REF: &'static i32 = &234;

    // @has - '//*[@id="associatedconstant.PTR"]' \
    //        'const PTR: *const u16 = _'
    pub const PTR: *const u16 = &90;

    // @has - '//*[@id="associatedconstant.ARR0"]' \
    //        'const ARR0: [u16; 8] = [1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080]'
    pub const ARR0: [u16; 8] = [12 * 90; 8];

    // Check that after a certain unspecified size threshold, array elements
    // won't be displayed anymore and that instead a *styled* series of ellipses is shown.
    // Re snapshot: Check that the series of ellipses is styled (has a certain CSS class).
    //
    // @has - '//*[@id="associatedconstant.ARR1"]' \
    //        'const ARR1: [u16; 100] = [………]'
    // @snapshot arr1 - '//*[@id="associatedconstant.ARR1"]//*[@class="code-header"]'
    pub const ARR1: [u16; 100] = [12; 52 + 50 - 2];

    // FIXME: We actually want to print the contents of slices!
    // @has - '//*[@id="associatedconstant.SLICE0"]' \
    //        "const SLICE0: &'static [bool] = _"
    pub const SLICE0: &'static [bool] = &[false, !true, true];

    //
    // The following two test cases are regression tests for issue #99630:
    // Make sure that we don't leak private and doc(hidden) struct fields
    // of cross-crate structs (i.e. structs from external crates).
    //

    // @has - '//*[@id="associatedconstant.DATA"]' \
    //        'const DATA: Data = Data { open: (0, 0, 1), .. }'
    // @snapshot data - '//*[@id="associatedconstant.DATA"]//*[@class="code-header"]'
    pub const DATA: aux::Data = aux::Data::new((0, 0, 1));

    // @has - '//*[@id="associatedconstant.OPAQ"]' \
    //        'const OPAQ: Opaque = Opaque(_)'
    // @snapshot opaq - '//*[@id="associatedconstant.OPAQ"]//*[@class="code-header"]'
    pub const OPAQ: aux::Opaque = aux::Opaque::new(0xff00);
}

pub struct TupStruct(#[doc(hidden)] pub (), pub i32, (), #[doc(hidden)] pub bool);

pub struct Container0 {
    #[doc(hidden)]
    pub hack: (),
}

pub struct Container1(Option<std::cell::Cell<u8>>);

pub type AdtSize = Size;

pub enum Size {
    Inhabited { inhabitants: u128 },
    Uninhabited,
    Unknown(Reason),
}

pub struct Reason;

use std::cmp::Ordering;

// @has 'consts/trait.Protocol.html'
pub trait Protocol {
    // Make sure that this formatting also applies to const exprs inside of trait items, not just
    // inside of inherent impl blocks or free constants.

    // @has - '//*[@id="associatedconstant.MATCH"]' \
    //        'const MATCH: u64 = 99'
    const MATCH: u64 = match 1 + 4 {
        SUPPORT => 99,
        _ => 0,
    };

    // Re snapshot: Verify that hyperlinks are created.
    //
    // @has - '//*[@id="associatedconstant.OPT"]' \
    //        'const OPT: Option<Option<Ordering>> = Some(Some(Equal))'
    // @snapshot opt - '//*[@id="associatedconstant.OPT"]//*[@class="code-header"]'
    const OPT: Option<Option<Ordering>> = Some(Some(Ordering::Equal));

    // Test that there is a depth limit. Subexpressions exceeding the maximum depth are
    // rendered as *styled* ellipses.
    // Re snapshot: Check that the ellipses are styled (have a certain CSS class).
    //
    // @has - '//*[@id="associatedconstant.DEEP0"]' \
    //        'const DEEP0: Option<Option<Option<Ordering>>> = Some(Some(Some(…)))'
    // @snapshot deep0 - '//*[@id="associatedconstant.DEEP0"]//*[@class="code-header"]'
    const DEEP0: Option<Option<Option<Ordering>>> = Some(Some(Some(Ordering::Equal)));

    // FIXME: Add more depth tests

    // @has - '//*[@id="associatedconstant.STR0"]' \
    //        "const STR0: &'static str = \"I want to <em>escape</em>!\""
    const STR0: &'static str = "I want to <em>escape</em>!";

    // Check that after a certain unspecified size threshold, the string contents
    // won't be displayed anymore and that instead a *styled* series of ellipses is shown.
    // Re snapshot: Check that the series of ellipses is styled (has a certain CSS class).
    //
    // @has - '//*[@id="associatedconstant.STR1"]' \
    //        "const STR1: &'static str = \"………\""
    // @snapshot str1 - '//*[@id="associatedconstant.STR1"]//*[@class="code-header"]'
    const STR1: &'static str = "\
        This is the start of a relatively long text. \
        I might as well throw some more words into it. \
        Informative content? Never heard of it! \
        That's probably one of the reasons why I shouldn't be included \
        into the generated documentation, don't you think so, too?\
    ";

    // @has - '//*[@id="associatedconstant.BYTE_STR0"]' \
    //        "const BYTE_STR0: &'static [u8] = b\"I want to <em>escape</em>!\""
    const BYTE_STR0: &'static [u8] = b"I want to <em>escape</em>!";

    // Check that after a certain unspecified size threshold, the byte string contents
    // won't be displayed anymore and that instead a *styled* series of ellipses is shown.
    // Re snapshot: Check that the series of ellipses is styled (has a certain CSS class).
    //
    // @has - '//*[@id="associatedconstant.BYTE_STR1"]' \
    //        "const BYTE_STR1: &'static [u8] = b\"………\""
    // @snapshot byte-str1 - '//*[@id="associatedconstant.BYTE_STR1"]//*[@class="code-header"]'
    const BYTE_STR1: &'static [u8] = b"\
        AGTC CCTG GAAT TACC AAAA AACA TCCA AGTC CTCT \
        AGTC CCTG TCCA AGTC CTCT GAAT TACC AAAA AACA \
        AGTC CCTG GAAT TACC AAAA GGGG GGGG AGTC GTTT \
        GGGG AACA TCCA AGTC CTCT AGTC CCTG GAAT TACC \
        AGTC AAAA GAAT TACC CGAG AACA TCCA AGTC CTCT \
        AGTC CCTG GAAT TACC TTCC AACA TCCA AGTC CTCT\
    ";

    // @has - '//*[@id="associatedconstant.BYTE_ARR0"]' \
    //        'const BYTE_ARR0: [u8; 12] = *b"DEREFERENCED"'
    const BYTE_ARR0: [u8; 12] = *b"DEREFERENCED";

    // @has - '//*[@id="associatedconstant.BYTE_ARR1"]' \
    //        "const BYTE_ARR1: [u8; 7] = *b\"MINCED\\x00\""
    const BYTE_ARR1: [u8; 7] = [b'M', b'I', b'N', b'C', b'E', b'D', b'\0'];
}

const SUPPORT: i32 = 5;

pub mod exhaustiveness {
    // @has 'consts/exhaustiveness/constant.EXHAUSTIVE_UNIT_STRUCT.html'
    // @has - '//*[@class="docblock item-decl"]//code' \
    //        'const EXHAUSTIVE_UNIT_STRUCT: ExhaustiveUnitStruct = ExhaustiveUnitStruct'
    pub const EXHAUSTIVE_UNIT_STRUCT: ExhaustiveUnitStruct = ExhaustiveUnitStruct;

    // @has 'consts/exhaustiveness/constant.EXHAUSTIVE_TUPLE_STRUCT.html'
    // @has - '//*[@class="docblock item-decl"]//code' \
    //        'const EXHAUSTIVE_TUPLE_STRUCT: ExhaustiveTupleStruct = ExhaustiveTupleStruct(())'
    pub const EXHAUSTIVE_TUPLE_STRUCT: ExhaustiveTupleStruct = ExhaustiveTupleStruct(());

    // @has 'consts/exhaustiveness/constant.EXHAUSTIVE_STRUCT.html'
    // @has - '//*[@class="docblock item-decl"]//code' \
    //        'const EXHAUSTIVE_STRUCT: ExhaustiveStruct = ExhaustiveStruct { inner: () }'
    pub const EXHAUSTIVE_STRUCT: ExhaustiveStruct = ExhaustiveStruct { inner: () };

    // @has 'consts/exhaustiveness/constant.NON_EXHAUSTIVE_UNIT_STRUCT.html'
    // @has - '//*[@class="docblock item-decl"]//code' \
    //        'const NON_EXHAUSTIVE_UNIT_STRUCT: NonExhaustiveUnitStruct = NonExhaustiveUnitStruct { .. }'
    pub const NON_EXHAUSTIVE_UNIT_STRUCT: NonExhaustiveUnitStruct = NonExhaustiveUnitStruct;

    // @has 'consts/exhaustiveness/constant.NON_EXHAUSTIVE_TUPLE_STRUCT.html'
    // @has - '//*[@class="docblock item-decl"]//code' \
    //        'const NON_EXHAUSTIVE_TUPLE_STRUCT: NonExhaustiveTupleStruct = NonExhaustiveTupleStruct((), ..)'
    pub const NON_EXHAUSTIVE_TUPLE_STRUCT: NonExhaustiveTupleStruct = NonExhaustiveTupleStruct(());

    // @has 'consts/exhaustiveness/constant.NON_EXHAUSTIVE_STRUCT.html'
    // @has - '//*[@class="docblock item-decl"]//code' \
    //        'const NON_EXHAUSTIVE_STRUCT: NonExhaustiveStruct = NonExhaustiveStruct { inner: (), .. }'
    pub const NON_EXHAUSTIVE_STRUCT: NonExhaustiveStruct = NonExhaustiveStruct { inner: () };

    // Assert that full ranges are literally rendered as `RangeFull` and not `..` to make sure that
    // `..` unambiguously means “omitted fields” in our pseudo-Rust expression syntax.
    // See the comment in `render_const_value` for more details.
    //
    // @has 'consts/exhaustiveness/constant.RANGE_FULL.html'
    // @has - '//*[@class="docblock item-decl"]//code' \
    //        'const RANGE_FULL: RangeFull = RangeFull'
    pub const RANGE_FULL: std::ops::RangeFull = ..;

    pub struct ExhaustiveUnitStruct;
    pub struct ExhaustiveTupleStruct(pub ());
    pub struct ExhaustiveStruct { pub inner: () }
    #[non_exhaustive] pub struct NonExhaustiveUnitStruct;
    #[non_exhaustive] pub struct NonExhaustiveTupleStruct(pub ());
    #[non_exhaustive] pub struct NonExhaustiveStruct { pub inner: () }
}
