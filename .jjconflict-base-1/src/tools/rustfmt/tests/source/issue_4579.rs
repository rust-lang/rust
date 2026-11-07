// rustfmt-hard_tabs: true

#[macro_export]
macro_rules! main {
	() => {
		#[spirv(fragment)]
		pub fn main_fs(
			mut out_color: ::spirv_std::storage_class::Output<Vec4>,
			#[spirv(descriptor_set = 1)]iChannelResolution: ::spirv_std::storage_class::UniformConstant<
				[::spirv_std::glam::Vec3A; 4],
			>,
		) {
		}
	};
}
