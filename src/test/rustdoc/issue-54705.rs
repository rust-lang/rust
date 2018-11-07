// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


pub trait ScopeHandle<'scope> {}



// @has issue_54705/struct.ScopeFutureContents.html
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//*/code' "impl<'scope, S> \
// Send for ScopeFutureContents<'scope, S> where S: Sync"
//
// @has - '//*[@id="synthetic-implementations-list"]/*[@class="impl"]//*/code' "impl<'scope, S> \
// Sync for ScopeFutureContents<'scope, S> where S: Sync"
pub struct ScopeFutureContents<'scope, S>
    where S: ScopeHandle<'scope>,
{
    dummy: &'scope S,
    this: Box<ScopeFuture<'scope, S>>,
}

struct ScopeFuture<'scope, S>
    where S: ScopeHandle<'scope>,
{
    contents: ScopeFutureContents<'scope, S>,
}

unsafe impl<'scope, S> Send for ScopeFuture<'scope, S>
    where S: ScopeHandle<'scope>,
{}
unsafe impl<'scope, S> Sync for ScopeFuture<'scope, S>
    where S: ScopeHandle<'scope>,
{}
