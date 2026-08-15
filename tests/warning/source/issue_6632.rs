macro_rules! impl_routes_and_health {
	($($feature:literal, $variant:ident),* $(,)?) => {
		impl EitherState {
			pub(crate) fn service_name(&self) -> &'static str {
				match self {
					$(
						#[cfg(feature = $feature)]
						Self::$variant(s) => s.service_name(),// BuildService::service_name(s.clone()),
					)*
				}
			}
		}
	};
}
