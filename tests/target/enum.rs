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
    First(LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONG, // comment
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
    }
}
