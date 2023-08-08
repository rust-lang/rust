// revisions: current next
//[next] compile-flags: -Ztrait-solver=next

struct Test {
    func: Box<dyn FnMut() + 'static>,
}

fn main() {
    let closure: Box<dyn Fn() + 'static> = Box::new(|| ());
    let test = Box::new(Test { func: closure }); //~ ERROR trait upcasting coercion is experimental [E0658]
}
