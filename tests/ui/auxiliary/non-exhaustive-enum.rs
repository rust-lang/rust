// Stripped down version of the ErrorKind enum of std
#[non_exhaustive]
pub enum ErrorKind {
    NotFound,
    PermissionDenied,
    #[doc(hidden)]
    Uncategorized,
}

#[non_exhaustive]
pub enum ExtNonExhaustiveEnum {
    Unit,
    Tuple(i32),
    Struct { field: i32 },
}

pub enum ExtNonExhaustiveVariant {
    ExhaustiveUnit,
    #[non_exhaustive]
    Unit,
    #[non_exhaustive]
    Tuple(i32),
    #[non_exhaustive]
    StructNoField {},
    #[non_exhaustive]
    Struct {
        field: i32,
    },
}
