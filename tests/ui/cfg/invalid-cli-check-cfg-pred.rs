//@ compile-flags: --check-cfg 'foo=1x'
//@ reference: cfg.option-spec
//@ reference: cfg.option-name
//@ reference: cfg.option-key-value

fn main() {}

//~? ERROR invalid `--check-cfg` argument
