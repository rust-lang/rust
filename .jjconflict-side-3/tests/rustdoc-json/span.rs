pub mod bar {}
// This test ensures that spans are 1-indexed.
//@ is "$.index[?(@.name=='span')].span.begin" "[1, 1]"
//@ is "$.index[?(@.name=='bar')].span.begin" "[1, 1]"
