//! Check that we properly gate unstable print requests (`--print=KIND`) and require the user to
//! specify `-Z unstable-options` to use unstable print requests.

// We don't care about the exact *stdout* output (i.e. what the print requests actually give back)
// for the purposes of this test.
//@ dont-check-compiler-stdout

// We want to check for the core error message of the unstable print requests being `-Z
// unstable-options`-gated and not the help because the help can change with addition of a new print
// request, which is not important for the purposes of this test.
//@ dont-check-compiler-stderr

// =======================
// Unstable print requests
// =======================

//@ revisions: all_target_specs_json
//@[all_target_specs_json] compile-flags: --print=all-target-specs-json

//@ revisions: crate_root_lint_levels
//@[crate_root_lint_levels] compile-flags: --print=crate-root-lint-levels

//@ revisions: check_cfg
//@[check_cfg] compile-flags: --print=check-cfg

//@ revisions: supported_crate_types
//@[supported_crate_types] compile-flags: --print=supported-crate-types

//@ revisions: target_spec_json
//@[target_spec_json] compile-flags: --print=target-spec-json

// =======================
// Stable print requests
// =======================

//@ revisions: calling_conventions
//@[calling_conventions] compile-flags: --print=calling-conventions
//@[calling_conventions] check-pass

//@ revisions: cfg
//@[cfg] compile-flags: --print=cfg
//@[cfg] check-pass

//@ revisions: code_models
//@[code_models] compile-flags: --print=code-models
//@[code_models] check-pass

//@ revisions: crate_name
//@[crate_name] compile-flags: --print=crate-name
//@[crate_name] check-pass

// Note: `--print=deployment_target` is only accepted on Apple targets.
//@ revisions: deployment_target
//@[deployment_target] only-apple
//@[deployment_target] compile-flags: --print=deployment-target
//@[deployment_target] check-pass

//@ revisions: file_names
//@[file_names] compile-flags: --print=file-names
//@[file_names] check-pass

//@ revisions: host_tuple
//@[host_tuple] compile-flags: --print=host-tuple
//@[host_tuple] check-pass

//@ revisions: link_args
//@[link_args] compile-flags: --print=link-args
//@[link_args] check-pass

//@ revisions: native_static_libs
//@[native_static_libs] compile-flags: --print=native-static-libs
//@[native_static_libs] check-pass

//@ revisions: relocation_models
//@[relocation_models] compile-flags: --print=relocation-models
//@[relocation_models] check-pass

//@ revisions: split_debuginfo
//@[split_debuginfo] compile-flags: --print=split-debuginfo
//@[split_debuginfo] check-pass

//@ revisions: stack_protector_strategies
//@[stack_protector_strategies] compile-flags: --print=stack-protector-strategies
//@[stack_protector_strategies] check-pass

//@ revisions: target_cpus
//@[target_cpus] compile-flags: --print=target-cpus
//@[target_cpus] check-pass

//@ revisions: target_features
//@[target_features] compile-flags: --print=target-features
//@[target_features] check-pass

//@ revisions: target_libdir
//@[target_libdir] compile-flags: --print=target-libdir
//@[target_libdir] check-pass

//@ revisions: target_list
//@[target_list] compile-flags: --print=target-list
//@[target_list] check-pass

//@ revisions: tls_models
//@[tls_models] compile-flags: --print=tls-models
//@[tls_models] check-pass

fn main() {}

//[all_target_specs_json]~? ERROR the `-Z unstable-options` flag must also be passed to enable the `all-target-specs-json` print option
//[crate_root_lint_levels]~? ERROR the `-Z unstable-options` flag must also be passed to enable the `crate-root-lint-levels` print option
//[check_cfg]~? ERROR the `-Z unstable-options` flag must also be passed to enable the `check-cfg` print option
//[supported_crate_types]~? ERROR the `-Z unstable-options` flag must also be passed to enable the `supported-crate-types` print option
//[target_spec_json]~? ERROR the `-Z unstable-options` flag must also be passed to enable the `target-spec-json` print option
