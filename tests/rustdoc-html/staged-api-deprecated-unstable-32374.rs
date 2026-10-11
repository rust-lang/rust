// https://github.com/rust-lang/rust/issues/32374
#![feature(staged_api)]
#![doc(issue_tracker_base_url = "https://issue_url/")]
#![unstable(feature = "test", issue = "32374")]
#![crate_name="issue_32374"]

//@ matches issue_32374/index.html '//dt/span[@class="stab deprecated"]' 'Deprecated'
//@ matches issue_32374/index.html '//dt/span[@class="stab unstable"]' 'Experimental'
//@ matches issue_32374/index.html '//dd/text()' 'Docs'

//@ has issue_32374/type.T.html '//*[@class="stab deprecated"]/span' '👎'
//@ has issue_32374/type.T.html '//*[@class="stab deprecated"]/span' \
//      'Deprecated since 1.0.0: text'
//@ hasraw - '<code>test</code>&nbsp;<a href="https://issue_url/32374">#32374</a>'
//@ matches issue_32374/type.T.html '//*[@class="stab unstable"]' '🔬'
//@ matches issue_32374/type.T.html '//*[@class="stab unstable"]' \
//     'This is a nightly-only experimental API. \(test\s#32374\)$'
/// Docs
#[deprecated(since = "1.0.0", note = "text")]
#[unstable(feature = "test", issue = "32374")]
pub struct T;

//@ has issue_32374/type.U.html '//*[@class="stab deprecated"]' '👎'
//@ has issue_32374/type.U.html '//*[@class="stab deprecated"]' \
//     'Deprecated since 1.0.0: deprecated'
//@ has issue_32374/type.U.html '//*[@class="stab unstable"]' '🔬'
//@ has issue_32374/type.U.html '//*[@class="stab unstable"]' \
//     'This is a nightly-only experimental API. (test #32374)'
#[deprecated(since = "1.0.0", note = "deprecated")]
#[unstable(feature = "test", issue = "32374", reason = "unstable")]
pub struct U;
