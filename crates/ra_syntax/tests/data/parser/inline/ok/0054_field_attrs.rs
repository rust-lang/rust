struct S {
    #[serde(with = "url_serde")]
    pub uri: Uri,
}
