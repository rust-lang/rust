//@ !hasraw crates.js 'dep_missing'
//@ !hasraw search.index/name/*.js 'DepMissing'
//@ has dep_missing/index.html
pub struct DepMissing;
