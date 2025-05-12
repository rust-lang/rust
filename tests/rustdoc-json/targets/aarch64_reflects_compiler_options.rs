//@ only-aarch64

// If we enable SVE Bit Permute, we should see that it is enabled
//@ compile-flags: -Ctarget-feature=+sve2-bitperm
//@ is "$.target.target_features[?(@.name=='sve2-bitperm')].globally_enabled" true

// As well as its dependency chain
//@ is "$.target.target_features[?(@.name=='sve2')].globally_enabled" true
//@ is "$.target.target_features[?(@.name=='sve')].globally_enabled" true
//@ is "$.target.target_features[?(@.name=='neon')].globally_enabled" true
