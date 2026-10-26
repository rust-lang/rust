//@ check-pass
// This is currently stable behavior, which was almost accidentally made an
// error in #102161 since there is no test exercising it. I am not sure if
// this _should_ be the desired behavior, but at least we should know if it
// changes.
// The `duplicate_parameter_names` lint now warns against it.

fn main() {}

trait Foo {
    fn fn_with_type_named_same_as_local_in_param(b: i32, b: i32);
    //~^ WARN identifier `b` is bound more than once in this parameter list
}
