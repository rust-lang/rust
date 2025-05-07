//@ only-x86_64

// If we enable AVX2, we should see that it is enabled
//@ compile-flags: -Ctarget-feature=+avx2
//@ is "$.target.target_features[?(@.name=='avx2')].globally_enabled" true

// As well as its dependency chain
//@ is "$.target.target_features[?(@.name=='avx')].globally_enabled" true
//@ is "$.target.target_features[?(@.name=='sse4.2')].globally_enabled" true
//@ is "$.target.target_features[?(@.name=='sse4.1')].globally_enabled" true
