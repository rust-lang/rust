//@ only-aarch64-apple-darwin

//@ is "$.target.triple" \"aarch64-apple-darwin\"
//@ is "$.target.target_features[?(@.name=='vh')].globally_enabled" true
//@ is "$.target.target_features[?(@.name=='sve')].globally_enabled" false
//@ has "$.target.target_features[?(@.name=='sve2')].implies_features" '["sve"]'
//@ is "$.target.target_features[?(@.name=='sve2')].unstable_feature_gate" null

// If this breaks due to stabilization, check rustc_target::target_features for a replacement
//@ is "$.target.target_features[?(@.name=='cssc')].unstable_feature_gate" '"aarch64_unstable_target_feature"'
//@ is "$.target.target_features[?(@.name=='v9a')].unstable_feature_gate" '"aarch64_ver_target_feature"'

// Ensure we don't look like x86-64
//@ !has "$.target.target_features[?(@.name=='avx2')]"
