// rustfmt-wrap_comments: true
// Enums test

#[atrr]
pub enum Test {
    A,
    B(u32, A /* comment */, SomeType),
    /// Doc comment
    C,
}

pub enum Foo<'a, Y: Baz>
where
    X: Whatever,
{
    A,
}

enum EmtpyWithComment {
    // Some comment
}

// C-style enum
enum Bar {
    A = 1,
    #[someAttr(test)]
    B = 2, // comment
    C,
}

enum LongVariants {
    First(
        LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONG, // comment
        VARIANT,
    ),
    // This is the second variant
    Second,
}

enum StructLikeVariants {
    Normal(u32, String),
    StructLike {
        x: i32, // Test comment
        // Pre-comment
        #[Attr50]
        y: SomeType, // Another Comment
    },
    SL {
        a: A,
    },
}

enum X {
    CreateWebGLPaintTask(
        Size2D<i32>,
        GLContextAttributes,
        IpcSender<Result<(IpcSender<CanvasMsg>, usize), String>>,
    ), // This is a post comment
}

pub enum EnumWithAttributes {
    //This is a pre comment
    // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    TupleVar(usize, usize, usize), /* AAAA AAAAAAAAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAAAAA
                                    * AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA */
    // Pre Comment
    #[rustfmt::skip]
    SkippedItem(String,String,), // Post-comment
    #[another_attr]
    #[attr2]
    ItemStruct {
        x: usize,
        y: usize,
    }, /* Comment AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA */
    // And another
    ForcedPreflight, /* AAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                      * AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA */
}

pub enum SingleTuple {
    // Pre Comment AAAAAAAAAAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    Match(usize, usize, String), /* Post-comment AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA */
}

pub enum SingleStruct {
    Match { name: String, loc: usize }, // Post-comment
}

pub enum GenericEnum<I, T>
where
    I: Iterator<Item = T>,
{
    // Pre Comment
    Left { list: I, root: T },  // Post-comment
    Right { list: I, root: T }, // Post Comment
}

enum EmtpyWithComment {
    // Some comment
}

enum TestFormatFails {
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
}

fn nested_enum_test() {
    if true {
        enum TestEnum {
            One(
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
            ), /* AAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAAAAAAAA
                * AAAAAAAAAAAAAAAAAAAAAA */
            Two, /* AAAAAAAAAAAAAAAAAA  AAAAAAAAAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                  * AAAAAAAAAAAAAAAAAA */
        }
        enum TestNestedFormatFail {
            AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
        }
    }
}

pub struct EmtpyWithComment {
    // FIXME: Implement this struct
}

// #1115
pub enum Bencoding<'i> {
    Str(&'i [u8]),
    Int(i64),
    List(Vec<Bencoding<'i>>),
    /// A bencoded dict value. The first element the slice of bytes in the
    /// source that the dict is composed of. The second is the dict, decoded
    /// into an ordered map.
    // TODO make Dict "structlike" AKA name the two values.
    Dict(&'i [u8], BTreeMap<&'i [u8], Bencoding<'i>>),
}

// #1261
pub enum CoreResourceMsg {
    SetCookieForUrl(
        ServoUrl,
        #[serde(
            deserialize_with = "::hyper_serde::deserialize",
            serialize_with = "::hyper_serde::serialize"
        )]
        Cookie,
        CookieSource,
    ),
}

enum Loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong
{}
enum Looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong
{}
enum Loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong
{}
enum Loooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooong
{
    Foo,
}

// #1046
pub enum Entry<'a, K: 'a, V: 'a> {
    Vacant(#[stable(feature = "rust1", since = "1.0.0")] VacantEntry<'a, K, V>),
    Occupied(#[stable(feature = "rust1", since = "1.0.0")] OccupiedEntry<'a, K, V>),
}

// #2081
pub enum ForegroundColor {
    CYAN =
        (winapi::FOREGROUND_INTENSITY | winapi::FOREGROUND_GREEN | winapi::FOREGROUND_BLUE) as u16,
}

// #2098
pub enum E<'a> {
    V(<std::slice::Iter<'a, Xxxxxxxxxxxxxx> as Iterator>::Item),
}

// #1809
enum State {
    TryRecv {
        pos: usize,
        lap: u8,
        closed_count: usize,
    },
    Subscribe {
        pos: usize,
    },
    IsReady {
        pos: usize,
        ready: bool,
    },
    Unsubscribe {
        pos: usize,
        lap: u8,
        id_woken: usize,
    },
    FinalTryRecv {
        pos: usize,
        id_woken: usize,
    },
    TimedOut,
    Disconnected,
}

// #2190
#[derive(Debug, Fail)]
enum AnError {
    #[fail(
        display = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )]
    UnexpectedSingleToken { token: syn::Token },
}

// #2193
enum WidthOf101 {
    #[fail(display = ".....................................................")]
    Io(::std::io::Error),
    #[fail(display = ".....................................................")]
    Ioo(::std::io::Error),
    Xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(::std::io::Error),
    Xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx(
        ::std::io::Error,
    ),
}

// #2389
pub enum QlError {
    #[fail(display = "Parsing error: {}", 0)]
    LexError(parser::lexer::LexError),
    #[fail(display = "Parsing error: {:?}", 0)]
    ParseError(parser::ParseError),
    #[fail(display = "Validation error: {:?}", 0)]
    ValidationError(Vec<validation::Error>),
    #[fail(display = "Execution error: {}", 0)]
    ExecutionError(String),
    // (from, to)
    #[fail(display = "Translation error: from {} to {}", 0, 1)]
    TranslationError(String, String),
    // (kind, input, expected)
    #[fail(
        display = "aaaaaaaaaaaaCould not find {}: Found: {}, expected: {:?}",
        0, 1, 2
    )]
    ResolveError(&'static str, String, Option<String>),
}

// #2594
enum Foo {}
enum Bar {}

// #3562
enum PublishedFileVisibility {
    Public =
        sys::ERemoteStoragePublishedFileVisibility_k_ERemoteStoragePublishedFileVisibilityPublic,
    FriendsOnly = sys::ERemoteStoragePublishedFileVisibility_k_ERemoteStoragePublishedFileVisibilityFriendsOnly,
    Private =
        sys::ERemoteStoragePublishedFileVisibility_k_ERemoteStoragePublishedFileVisibilityPrivate,
}

// #3771
//#![feature(arbitrary_enum_discriminant)]
#[repr(u32)]
pub enum E {
    A {
        a: u32,
    } = 0x100,
    B {
        field1: u32,
        field2: u8,
        field3: m::M,
    } = 0x300, // comment
}
