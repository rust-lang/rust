// revisions: e2015 e2018 e2021
//[e2018] compile-flags: --edition=2018
//[e2021] compile-flags: --edition=2021
// run-pass

#![feature(panic_info_message)]

fn main() {
    std::panic::set_hook(Box::new(|info| assert_eq!(info.message().as_str().unwrap(), "cake")));
    let _ = std::panic::catch_unwind(|| panic!("cake"));
}
