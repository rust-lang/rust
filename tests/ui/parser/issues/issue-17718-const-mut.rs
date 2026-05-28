const
mut //~ ERROR: const globals cannot be mutable
//~^^ HELP you might want to declare a static instead
FOO: usize = 3;

fn main() {
}
