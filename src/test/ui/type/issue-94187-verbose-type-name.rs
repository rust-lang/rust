// Check to insure that the output of `std::any::type_name` does not change based on -Zverbose
// when printing constants
// run-pass
// edition: 2018
// revisions: normal verbose
// [verbose]compile-flags:-Zverbose

struct Wrapper<const VALUE: usize>;

fn main() {
    assert_eq!(std::any::type_name::<[u32; 0]>(), "[u32; 0]");
    assert_eq!(std::any::type_name::<Wrapper<0>>(), "issue_94187_verbose_type_name::Wrapper<0>");
}
