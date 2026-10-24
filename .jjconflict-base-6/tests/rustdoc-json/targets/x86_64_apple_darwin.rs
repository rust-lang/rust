//@ only-x86_64-apple-darwin

//@ is "$.target.triple" \"x86_64-apple-darwin\"
//@ is "$.target.target_features[?(@.name=='sse2')].globally_enabled" true
//@ is "$.target.target_features[?(@.name=='avx2')].globally_enabled" false
//@ has "$.target.target_features[?(@.name=='avx2')].implies_features" '["avx"]'
//@ is "$.target.target_features[?(@.name=='avx2')].unstable_feature_gate" null

// If this breaks due to stabilization, check rustc_target::target_features for a replacement
//@ is "$.target.target_features[?(@.name=='amx-tile')].unstable_feature_gate" '"x86_amx_intrinsics"'
//@ is "$.target.target_features[?(@.name=='x87')].unstable_feature_gate" '"x87_target_feature"'

// Ensure we don't look like aarch64
//@ !has "$.target.target_features[?(@.name=='sve2')]"
