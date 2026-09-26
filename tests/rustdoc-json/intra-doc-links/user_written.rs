//! For motivation, see [the reasons](foo#reasons)

/// # Reasons
/// To test rustdoc json
pub fn foo() {}

//@ set foo = "$.index[?(@.name=='foo')].id"
//@ is "$.index[?(@.name=='user_written')].docs[0].links['foo#reasons']" $foo
