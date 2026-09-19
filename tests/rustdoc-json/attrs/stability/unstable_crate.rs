#![feature(staged_api)]
#![unstable(feature = "unstable_crate_feature", issue = "none")]

//@ is "$.index[?(@.name=='unstable_crate')].stability.level" '"unstable"'
//@ is "$.index[?(@.name=='unstable_crate')].stability.feature" '"unstable_crate_feature"'
//@ !has "$.index[?(@.name=='unstable_crate')].stability.since"
