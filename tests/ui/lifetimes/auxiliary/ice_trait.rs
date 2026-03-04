pub trait ExternalTrait {
    fn build_request(&mut self) -> impl std::future::Future<Output = ()>;
}
