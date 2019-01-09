#![feature(staged_api)]
#![doc(issue_tracker_base_url = "http://issue_url/")]

#![unstable(feature="test", issue="27759")]

// @has issue_27759/unstable/index.html
// @has - '<code>test</code>&nbsp;<a href="http://issue_url/27759">#27759</a>'
#[unstable(feature="test", issue="27759")]
pub mod unstable {
    // @has issue_27759/unstable/fn.issue.html
    // @has - '<code>test_function</code>&nbsp;<a href="http://issue_url/12345">#12345</a>'
    #[unstable(feature="test_function", issue="12345")]
    pub fn issue() {}
}
