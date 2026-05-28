//@ run-pass
// ^-- The above is needed as this issue is related to LLVM/codegen.

fn main() {
    type_error(|x| &x);
}

fn type_error<T>(
    _selector: for<'a> fn(&'a Vec<Box<dyn for<'b> Fn(&'b u8)>>) -> &'a Vec<Box<dyn Fn(T)>>,
) {
}
