// @has issue_41783/struct.Foo.html
// @!hasraw - 'space'
// @!hasraw - 'comment'
// @hasraw - '# <span class="ident">single'
// @hasraw - '## <span class="ident">double</span>'
// @hasraw - '### <span class="ident">triple</span>'
// @hasraw - '<span class="attribute">#[<span class="ident">outer</span>]</span>'
// @hasraw - '<span class="attribute">#![<span class="ident">inner</span>]</span>'

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
