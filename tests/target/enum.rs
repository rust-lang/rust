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
    where X: Whatever
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
    First(LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONG, // comment
          VARIANT),
    // This is the second variant
    Second,
}

enum StructLikeVariants {
    Normal(u32, String),
    StructLike {
        x: i32, // Test comment
        // Pre-comment
        #[Attr50]
        y: SomeType, // Aanother Comment
    },
    SL {
        a: A,
    },
}

enum X {
    CreateWebGLPaintTask(Size2D<i32>,
                         GLContextAttributes,
                         IpcSender<Result<(IpcSender<CanvasMsg>, usize), String>>), /* This is
                                                                                     * a post c
                                                                                     * omment */
}

pub enum EnumWithAttributes {
    // This is a pre comment
    // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    TupleVar(usize, usize, usize), /* AAAA AAAAAAAAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAAAAA
                                    * AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA */
    // Pre Comment
    #[rustfmt_skip]
    SkippedItem(String,String,), // Post-comment
    #[another_attr]
    #[attr2]
    ItemStruct {
        x: usize,
        y: usize,
    }, /* Comment AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        * AAAAAAAAAAAAAAAAAAA */
    // And another
    ForcedPreflight, /* AAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                      * AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA */
}

pub enum SingleTuple {
    // Pre Comment AAAAAAAAAAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    Match(usize, usize, String), /* Post-comment
                                  * AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                                  * A */
}

pub enum SingleStruct {
    Match {
        name: String,
        loc: usize,
    }, // Post-comment
}

pub enum GenericEnum<I, T>
    where I: Iterator<Item = T>
{
    // Pre Comment
    Left {
        list: I,
        root: T,
    }, // Post-comment
    Right {
        list: I,
        root: T,
    }, // Post Comment
}


enum EmtpyWithComment {
    // Some comment
}

enum TestFormatFails {
    AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
}

fn nested_enum_test() {
    if true {
        enum TestEnum {
            One(usize,
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
                usize), /* AAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAAAAAAAA
                         * AAAAAAAAAAAAAAAAAAAAAA */
            Two, /* AAAAAAAAAAAAAAAAAA  AAAAAAAAAAAAAAAAAAAAAA AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                  * AAAAAAAAAAAAAAAAAA */
        }
        enum TestNestedFormatFail {
            AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        }
    }
}

pub struct EmtpyWithComment {
    // FIXME: Implement this struct
}
