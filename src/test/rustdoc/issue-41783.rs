// @has issue_41783/struct.Foo.html
// @has - '//span[@class="hidden"]' 'space'
// @has - '//span[@class="hidden"]' 'comment'
// @has - '# <span class="ident">single'
// @has - '## <span class="ident">double</span>'
// @has - '### <span class="ident">triple</span>'
// @has - '<span class="attribute">#[<span class="ident">outer</span>]</span>'
// @has - '<span class="attribute">#![<span class="ident">inner</span>]</span>'

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
