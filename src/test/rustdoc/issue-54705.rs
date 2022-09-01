pub trait ScopeHandle<'scope> {}

// @has issue_54705/struct.ScopeFutureContents.html
// @has - '//*[@id="synthetic-implementations-list"]//*[@class="impl has-srclink"]//h3[@class="code-header in-band"]' \
// "impl<'scope, S> Send for ScopeFutureContents<'scope, S>where S: Sync"
//
// @has - '//*[@id="synthetic-implementations-list"]//*[@class="impl has-srclink"]//h3[@class="code-header in-band"]' \
// "impl<'scope, S> Sync for ScopeFutureContents<'scope, S>where S: Sync"
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
