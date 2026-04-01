// See https://github.com/rust-lang/rust/issues/101105

//@ !has "$.index[?(@.name=='nucleus')]"
mod corpus {
    pub mod nucleus {}
}
