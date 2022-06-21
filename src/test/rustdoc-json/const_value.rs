// Testing the formatting of constant values (i.e. evaluated constant expressions)
// where the specific format was first proposed in issue #98929.

// ignore-tidy-linelength
// edition:2021
// aux-crate:data=data.rs

// @has const_value.json

// Check that constant expressions are printed in their evaluated form.
//
// @is - "$.index[*][?(@.name=='HOUR_IN_SECONDS')].kind" \"constant\"
// @is - "$.index[*][?(@.name=='HOUR_IN_SECONDS')].inner.value" \"3600\"
pub const HOUR_IN_SECONDS: u64 = 60 * 60;

// @is - "$.index[*][?(@.name=='NEGATIVE')].kind" \"constant\"
// @is - "$.index[*][?(@.name=='NEGATIVE')].inner.value" \"-3600\"
pub const NEGATIVE: i64 = -60 * 60;

// @is - "$.index[*][?(@.name=='CONCATENATED')].kind" \"constant\"
// @is - "$.index[*][?(@.name=='CONCATENATED')].inner.value" '"\"[0, +∞)\""'
pub const CONCATENATED: &str = concat!("[", stringify!(0), ", ", "+∞", ")");

pub struct Record<'r> {
    pub one: &'r str,
    pub two: (i32,),
}

// Test that structs whose fields are all public and 1-tuples are displayed correctly.
// Furthermore, the struct fields should appear in definition order.
//
// @is - "$.index[*][?(@.name=='REC')].kind" \"constant\"
// @is - "$.index[*][?(@.name=='REC')].inner.value" '"Record { one: \"thriving\", two: (180,) }"'
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
//
// @is - "$.index[*][?(@.name=='STRUCT')].kind" \"constant\"
// @is - "$.index[*][?(@.name=='STRUCT')].inner.value" '"Struct { public: (), .. }"'
pub const STRUCT: Struct = Struct {
    private : /* SourceMap::span_to_snippet trap */ (),
    public: { 1 + 3; },
    hidden: ()
};

// Test that enum variants, 2-tuples, bools and structs (with private and doc(hidden) fields) nested
// within are rendered correctly. Further, check that there is a maximum depth.
//
// @is - "$.index[*][?(@.name=='NESTED')].kind" \"constant\"
// @is - "$.index[*][?(@.name=='NESTED')].inner.value" '"Some((Struct { public: …, .. }, false))"'
pub const NESTED: Option<(Struct, bool)> = Some((
    Struct {
        public: (),
        private: (),
        hidden: (),
    },
    false,
));

use std::sync::atomic::AtomicBool;

pub struct Struct {
    private: (),
    pub public: (),
    #[doc(hidden)]
    pub hidden: (),
}

impl Struct {
    // Check that even inside inherent impl blocks private and doc(hidden) struct fields
    // are not displayed.
    //
    // @is - "$.index[*][?(@.name=='SELF')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='SELF')].inner.default" '"Struct { public: (), .. }"'
    pub const SELF: Self = Self {
        private: (),
        public: match () {
            () => {}
        },
        hidden: (),
    };

    // Verify that private and doc(hidden) *tuple* struct fields are not shown.
    // In their place, an underscore should be rendered.
    //
    // @is - "$.index[*][?(@.name=='TUP_STRUCT')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='TUP_STRUCT')].inner.default" '"TupStruct(_, -45, _, _)"'
    pub const TUP_STRUCT: TupStruct = TupStruct((), -45, (), false);

    // Check that structs whose fields are all doc(hidden) are rendered correctly.
    //
    // @is - "$.index[*][?(@.name=='SEALED0')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='SEALED0')].inner.default" '"Container0 { .. }"'
    pub const SEALED0: Container0 = Container0 { hack: () };

    // Check that *tuple* structs whose fields are all private are rendered correctly.
    //
    // @is - "$.index[*][?(@.name=='SEALED1')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='SEALED1')].inner.default" '"Container1(_)"'
    pub const SEALED1: Container1 = Container1(None);

    // Verify that cross-crate structs are displayed correctly and that their fields
    // are not leaked.
    //
    // @is - "$.index[*][?(@.name=='SEALED2')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='SEALED2')].inner.default" '"AtomicBool { .. }"'
    pub const SEALED2: AtomicBool = AtomicBool::new(true);

    // Test that (local) *unit* enum variants are rendered properly.
    //
    // @is - "$.index[*][?(@.name=='SUM0')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='SUM0')].inner.default" '"Uninhabited"'
    pub const SUM0: Size = self::Size::Uninhabited;

    // Test that (local) *struct* enum variants are rendered properly.
    //
    // @is - "$.index[*][?(@.name=='SUM1')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='SUM1')].inner.default" '"Inhabited { inhabitants: 9000 }"'
    pub const SUM1: Size = AdtSize::Inhabited { inhabitants: 9_000 };

    // Test that (local) *tuple* enum variants are rendered properly.
    //
    // @is - "$.index[*][?(@.name=='SUM2')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='SUM2')].inner.default" '"Unknown(Reason)"'
    pub const SUM2: Size = Size::Unknown(Reason);

    // @is - "$.index[*][?(@.name=='INT')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='INT')].inner.default" '"2368"'
    pub const INT: i64 = 2345 + 23;

    // @is - "$.index[*][?(@.name=='STR0')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='STR0')].inner.default" '"\"hello friends >.<\""'
    pub const STR0: &'static str = "hello friends >.<";

    // @is - "$.index[*][?(@.name=='FLOAT0')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='FLOAT0')].inner.default" '"2930.21997"'
    pub const FLOAT0: f32 = 2930.21997;

    // @is - "$.index[*][?(@.name=='FLOAT1')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='FLOAT1')].inner.default" '"-3.42E+21"'
    pub const FLOAT1: f64 = -3.42e+21;

    // FIXME: Should we attempt more sophisticated formatting for references?
    //
    // @is - "$.index[*][?(@.name=='REF')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='REF')].inner.default" '"_"'
    pub const REF: &'static i32 = &234;

    // FIXME: Should we attempt more sophisticated formatting for raw pointers?
    //
    // @is - "$.index[*][?(@.name=='PTR')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='PTR')].inner.default" '"_"'
    pub const PTR: *const u16 = &90;

    // @is - "$.index[*][?(@.name=='ARR0')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='ARR0')].inner.default" '"[1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080]"'
    pub const ARR0: [u16; 8] = [12 * 90; 8];

    // Check that after a certain unspecified size threshold, array elements
    // won't be displayed anymore and that instead a series of ellipses is shown.
    //
    // @is - "$.index[*][?(@.name=='ARR1')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='ARR1')].inner.default" '"[………]"'
    pub const ARR1: [u16; 100] = [12; 52 + 50 - 2];

    // FIXME: We actually want to print the contents of slices!
    // @is - "$.index[*][?(@.name=='SLICE0')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='SLICE0')].inner.default" '"_"'
    pub const SLICE0: &'static [bool] = &[false, !true, true];

    //
    // Make sure that we don't leak private and doc(hidden) struct fields
    // of cross-crate structs (i.e. structs from external crates).
    //

    // @is - "$.index[*][?(@.name=='DATA')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='DATA')].inner.default" '"Data { open: (0, 0, 1), .. }"'
    pub const DATA: data::Data = data::Data::new((0, 0, 1));

    // @is - "$.index[*][?(@.name=='OPAQ')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='OPAQ')].inner.default" '"Opaque(_)"'
    pub const OPAQ: data::Opaque = data::Opaque::new(0xff00);
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

pub trait Protocol {
    // Make sure that this formatting also applies to const exprs inside of trait items, not just
    // inside of inherent impl blocks or free constants.

    // @is - "$.index[*][?(@.name=='MATCH')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='MATCH')].inner.default" '"99"'
    const MATCH: u64 = match 1 + 4 {
        SUPPORT => 99,
        _ => 0,
    };

    // @is - "$.index[*][?(@.name=='OPT')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='OPT')].inner.default" '"Some(Some(Equal))"'
    const OPT: Option<Option<Ordering>> = Some(Some(Ordering::Equal));

    // Test that there is a depth limit. Subexpressions exceeding the maximum depth are
    // rendered as ellipses.
    //
    // @is - "$.index[*][?(@.name=='DEEP0')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='DEEP0')].inner.default" '"Some(Some(Some(…)))"'
    const DEEP0: Option<Option<Option<Ordering>>> = Some(Some(Some(Ordering::Equal)));

    // FIXME: Add more depth tests

    // Check that after a certain unspecified size threshold, the string contents
    // won't be displayed anymore and that instead a series of ellipses is shown.
    //
    // @is - "$.index[*][?(@.name=='STR1')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='STR1')].inner.default" '"\"………\""'
    const STR1: &'static str = "\
        This is the start of a relatively long text. \
        I might as well throw some more words into it. \
        Informative content? Never heard of it! \
        That's probably one of the reasons why I shouldn't be included \
        into the generated documentation, don't you think so, too?\
    ";

    // @is - "$.index[*][?(@.name=='BYTE_STR0')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='BYTE_STR0')].inner.default" '"b\"Stuck in the days of yore! >.<\""'
    const BYTE_STR0: &'static [u8] = b"Stuck in the days of yore! >.<";

    // Check that after a certain unspecified size threshold, the byte string contents
    // won't be displayed anymore and that instead a series of ellipses is shown.
    //
    // @is - "$.index[*][?(@.name=='BYTE_STR1')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='BYTE_STR1')].inner.default" '"b\"………\""'
    const BYTE_STR1: &'static [u8] = b"\
        AGTC CCTG GAAT TACC AAAA AACA TCCA AGTC CTCT \
        AGTC CCTG TCCA AGTC CTCT GAAT TACC AAAA AACA \
        AGTC CCTG GAAT TACC AAAA GGGG GGGG AGTC GTTT \
        GGGG AACA TCCA AGTC CTCT AGTC CCTG GAAT TACC \
        AGTC AAAA GAAT TACC CGAG AACA TCCA AGTC CTCT \
        AGTC CCTG GAAT TACC TTCC AACA TCCA AGTC CTCT\
    ";

    // @is - "$.index[*][?(@.name=='BYTE_ARR0')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='BYTE_ARR0')].inner.default" '"*b\"DEREFERENCED\""'
    const BYTE_ARR0: [u8; 12] = *b"DEREFERENCED";

    // @is - "$.index[*][?(@.name=='BYTE_ARR1')].kind" \"assoc_const\"
    // @is - "$.index[*][?(@.name=='BYTE_ARR1')].inner.default" '"*b\"MINCED\\x00\""'
    const BYTE_ARR1: [u8; 7] = [b'M', b'I', b'N', b'C', b'E', b'D', b'\0'];
}

const SUPPORT: i32 = 5;
