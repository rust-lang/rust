// @generated
impl AnyProvider for BakedDataProvider {
    fn load_any(&self, key: DataKey, req: DataRequest) -> Result<AnyResponse, DataError> {
        const ANDLISTV1MARKER: ::icu_provider::DataKeyHash =
            ::icu_list::provider::AndListV1Marker::KEY.hashed();
        #[allow(clippy::match_single_binding)]
        match key.hashed() {
            ANDLISTV1MARKER => list::and_v1::DATA
                .get_by(|k| req.locale.strict_cmp(k.as_bytes()).reverse())
                .copied()
                .map(AnyPayload::from_static_ref)
                .ok_or(DataErrorKind::MissingLocale),
            _ => Err(DataErrorKind::MissingDataKey),
        }
        .map_err(|e| e.with_req(key, req))
        .map(|payload| AnyResponse { payload: Some(payload), metadata: Default::default() })
    }
}
