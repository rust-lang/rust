// ignore-windows
// ignore-sgx std::os::fortanix_sgx::usercalls::alloc::Iter changes compiler suggestions
// compile-flags: --error-format pretty-json --json=diagnostic-rendered-ansi

// The output for humans should just highlight the whole span without showing
// the suggested replacement, but we also want to test that suggested
// replacement only removes one set of parentheses, rather than naïvely
// stripping away any starting or ending parenthesis characters—hence this
// test of the JSON error format.

fn main() {
    let x: Iter;
}
