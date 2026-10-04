// Check that type privacy is taken into account when considering reachability

//@ check-pass

#![feature(decl_macro, staged_api)]
#![stable(feature = "test", since = "1.0.0")]

// Type privacy should prevent use of these in other crates, so we shouldn't
// need a stability annotation.
fn private_function() {}
struct PrivateStruct { f: () }
enum PrivateEnum { V }
union PrivateUnion { g: () }
trait PrivateTrait {}

#[stable(feature = "test", since = "1.0.0")]
pub macro m() {}

fn main() {}
