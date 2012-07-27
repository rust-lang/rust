
// check that the &int here does not cause us to think that `foo`
// contains region pointers
enum foo = fn~(x: &int);

fn take_foo(x: foo/&) {} //~ ERROR no region bound is allowed on `foo`

fn main() {
}