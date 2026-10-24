//@ is "$.index[?(@.name=='A')].attrs" '[{"other": "#[default]"}]'
#[derive(Default)]
pub enum Test {
    #[default]
    A,
    B,
}
