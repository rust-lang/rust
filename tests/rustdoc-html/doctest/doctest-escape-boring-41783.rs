// https://github.com/rust-lang/rust/issues/41783
#![crate_name="foo"]

//@ has foo/struct.Foo.html
//@ !hasraw - 'space'
//@ !hasraw - 'comment'
//@ hasraw - '<span class="attr">#[outer]'
//@ !hasraw - '<span class="attr">#[outer]</span>'
//@ hasraw - '#![inner]</span>'
//@ !hasraw - '<span class="attr">#![inner]</span>'
//@ snapshot 'codeblock' - '//*[@class="toggle top-doc"]/*[@class="docblock"]//pre/code'

/// ```no_run
/// # # space
/// # comment
/// ## single
/// ### double
/// #### triple
/// ##[outer]
/// ##![inner]
/// ```
pub struct Foo;
