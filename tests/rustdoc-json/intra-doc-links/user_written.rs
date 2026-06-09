//! For motivation, see [the reasons](foo#reasons)

/// # Reasons
/// To test rustdoc json
pub fn foo() {}

//@ set foo = "$.index[?(@.name=='foo')].id"
//@ is "$.index[?(@.name=='user_written')].links['foo#reasons']" $foo
