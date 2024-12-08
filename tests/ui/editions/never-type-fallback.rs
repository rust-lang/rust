//@ revisions: e2021 e2024
//
//@[e2021] edition: 2021
//@[e2024] edition: 2024
//
//@ run-pass
//@ check-run-results

fn main() {
    print_return_type_of(|| panic!());
}

fn print_return_type_of<R>(_: impl FnOnce() -> R) {
    println!("return type = {}", std::any::type_name::<R>());
}
