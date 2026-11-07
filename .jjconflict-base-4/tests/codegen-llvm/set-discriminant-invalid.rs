//@ compile-flags: -C opt-level=0
#![crate_type = "lib"]

pub enum ApiError {}
#[allow(dead_code)]
pub struct TokioError {
    b: bool,
}
pub enum Error {
    Api { source: ApiError },
    Ethereum,
    Tokio { source: TokioError },
}
struct Api;
impl IntoError<Error> for Api {
    type Source = ApiError;
    // CHECK-LABEL: @into_error
    // CHECK: llvm.trap()
    // Also check the next instruction to make sure we do not match against `trap`
    // elsewhere in the code.
    // CHECK-NEXT: ret i8 poison
    #[no_mangle]
    fn into_error(self, error: Self::Source) -> Error {
        Error::Api { source: error }
    }
}

pub trait IntoError<E> {
    /// The underlying error
    type Source;

    /// Combine the information to produce the error
    fn into_error(self, source: Self::Source) -> E;
}
