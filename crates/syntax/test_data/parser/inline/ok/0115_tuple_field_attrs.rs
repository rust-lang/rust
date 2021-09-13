struct S (
    #[serde(with = "url_serde")]
    pub Uri,
);

enum S {
    Uri(#[serde(with = "url_serde")] Uri),
}
