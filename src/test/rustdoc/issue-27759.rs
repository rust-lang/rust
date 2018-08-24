#![feature(staged_api)]
#![doc(issue_tracker_base_url = "http://issue_url/")]

#![unstable(feature="test", issue="27759")]

// @has issue_27759/unstable/index.html
// @has - '<code>test </code>'
// @has - '<a href="http://issue_url/27759">#27759</a>'
#[unstable(feature="test", issue="27759")]
pub mod unstable {
    // @has issue_27759/unstable/fn.issue.html
    // @has - '<code>test_function </code>'
    // @has - '<a href="http://issue_url/1234567890">#1234567890</a>'
    #[unstable(feature="test_function", issue="1234567890")]
    pub fn issue() {}
}
