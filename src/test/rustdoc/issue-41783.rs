// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// @has issue_41783/struct.Foo.html
// @!has - 'space'
// @!has - 'comment'
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
