const
mut //~ ERROR: const globals cannot be mutable
//~^ HELP did you mean to declare a static?
FOO: usize = 3;

fn main() {
}
