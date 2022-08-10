// @has issue_41783/struct.Foo.html
// @!has - 'space'
// @!has - 'comment'
// @hastext - '# <span class="ident">single'
// @hastext - '## <span class="ident">double</span>'
// @hastext - '### <span class="ident">triple</span>'
// @hastext - '<span class="attribute">#[<span class="ident">outer</span>]</span>'
// @hastext - '<span class="attribute">#![<span class="ident">inner</span>]</span>'

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
