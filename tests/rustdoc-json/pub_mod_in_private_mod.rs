// See https://github.com/rust-lang/rust/issues/101105

//@ jq_count '.index[] | select(.name == "nucleus")' 0
mod corpus {
    pub mod nucleus {}
}
