use std::pin::Pin;
use std::future::Future;

pub trait Service<S, Request>: Sized + Send + Sync + 'static {
    /// The type of response returned by the service.
    type Response: Send + 'static;

    /// The type of error returned by the service.
    type Error: Send + 'static;

    /// Serve a response or error for the given request,
    /// using the given context.
    fn serve(
        &self,
        ctx: Context<S>,
        req: Request,
    ) -> impl Future<Output = Result<Self::Response, Self::Error>> + Send + '_;
}

pub struct Context<S> {
    state: S,
}
