/// Async closure in an external crate — required because decoder.rs is only called for
/// defs loaded from .rmeta files, not local crate defs.
pub fn returns_async_closure() -> impl futures::Sink<(), Error = std::io::Error> {
    futures::sink::unfold((), async |(), ()| Ok::<_, std::io::Error>(()))
}
