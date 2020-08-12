struct S (
    #[serde(with = "url_serde")]
    pub Uri,
);
