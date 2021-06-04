impl !Display for JoinHandle {}

impl !Box<JoinHandle> {}

impl !std::fmt::Display
    for JoinHandle<T: std::future::Future + std::marker::Send + std::marker::Sync>
{
}

impl
    !JoinHandle<T: std::future::Future<Output> + std::marker::Send + std::marker::Sync + 'static>
        + 'static
{
}
