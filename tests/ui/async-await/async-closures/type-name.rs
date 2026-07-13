//@ run-pass
//@ edition: 2024
// ignore-tidy-linelength

fn once<F: FnOnce() -> T, T>(f: F) -> T {
    f()
}

fn main() {
    let closure = async || {};

    // Name of future when called normally.
    let name = std::any::type_name_of_val(&closure());
    assert_eq!(name, "core::ops::coroutine::adapters::CoroutineFuture<type_name::main::{{closure}}::{{closure}}>");

    // Name of future when closure is called via its FnOnce shim.
    let name = std::any::type_name_of_val(&once(closure));
    assert_eq!(name, "core::ops::coroutine::adapters::CoroutineFuture<type_name::main::{{closure}}::{{closure}}::{{call_once}}>");
}
