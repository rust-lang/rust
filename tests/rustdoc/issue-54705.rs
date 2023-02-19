pub trait ScopeHandle<'scope> {}

// @has issue_54705/struct.ScopeFutureContents.html
// @has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<'scope, S> Send for ScopeFutureContents<'scope, S>where S: Sync"
//
// @has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
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
