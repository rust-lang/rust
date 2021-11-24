#![feature(staged_api)]
#![doc(issue_tracker_base_url = "https://issue_url/")]
#![unstable(feature = "test", issue = "32374")]

// @matches issue_32374/index.html '//*[@class="item-left unstable deprecated module-item"]/span[@class="stab deprecated"]' \
//      'Deprecated'
// @matches issue_32374/index.html '//*[@class="item-left unstable deprecated module-item"]/span[@class="stab unstable"]' \
//      'Experimental'
// @matches issue_32374/index.html '//*[@class="item-right docblock-short"]/text()' 'Docs'

// @has issue_32374/struct.T.html '//*[@class="stab deprecated"]' \
//      '👎 Deprecated since 1.0.0: text'
// @has - '<code>test</code>&nbsp;<a href="https://issue_url/32374">#32374</a>'
// @matches issue_32374/struct.T.html '//*[@class="stab unstable"]' \
//      '🔬 This is a nightly-only experimental API. \(test\s#32374\)$'
/// Docs
#[rustc_deprecated(since = "1.0.0", reason = "text")]
#[unstable(feature = "test", issue = "32374")]
pub struct T;

// @has issue_32374/struct.U.html '//*[@class="stab deprecated"]' \
//      '👎 Deprecated since 1.0.0: deprecated'
// @has issue_32374/struct.U.html '//*[@class="stab unstable"]' \
//      '🔬 This is a nightly-only experimental API. (test #32374)'
#[rustc_deprecated(since = "1.0.0", reason = "deprecated")]
#[unstable(feature = "test", issue = "32374", reason = "unstable")]
pub struct U;
