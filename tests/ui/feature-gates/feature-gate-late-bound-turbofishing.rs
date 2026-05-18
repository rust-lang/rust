// TODO: repalce E0794 with E0658?

// 'a is late bound
fn foo<'a>(b: &'a u32) -> &'a u32 { b }

fn main() {
    // error
    let f /* : FooFnItem<????> */ = foo::<'static>;
    //~^ ERROR
}
