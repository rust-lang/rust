// build-pass
// compile-flags: --crate-type lib

// Regression test for ICE which occurred when const propagating an enum whose discriminant
// niche triggered an integer underflow conmupting a delta.

pub enum ApiError {}
#[allow(dead_code)]
pub struct TokioError {
    b: bool,
}
pub enum Error {
    Api {
        source: ApiError,
    },
    Ethereum,
    Tokio {
        source: TokioError,
    },
}
struct Api;
impl IntoError<Error> for Api
{
    type Source = ApiError;
    fn into_error(self, error: Self::Source) -> Error {
        Error::Api {
            source: (|v| v)(error),
        }
    }
}

pub trait IntoError<E>
{
    /// The underlying error
    type Source;

    /// Combine the information to produce the error
    fn into_error(self, source: Self::Source) -> E;
}
