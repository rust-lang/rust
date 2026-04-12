pub trait ExternalTrait { fn build_request<'b>(&'b self) -> impl std::future::Future<Output = ()>; }
