mod repro {
    pub fn push() -> Result<(), ()> {
        self.api.map_api_result(|api| {
            #[allow(deprecated)]
            match api.apply_extrinsic_before_version_4_with_context()? {}
        })
    }
}
