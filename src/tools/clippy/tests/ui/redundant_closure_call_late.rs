// non rustfixable, see redundant_closure_call_fixable.rs
#![expect(unused_assignments)]

fn main() {
    #[expect(unused_variables)]
    let mut i = 1;

    // don't lint here, the closure is used more than once
    let closure = |i| i + 1;
    i = closure(3);
    i = closure(4);

    // lint here
    let redun_closure = || 1;
    i = redun_closure();
    //~^ redundant_closure_call

    // shadowed closures are supported, lint here
    let shadowed_closure = || 1;
    i = shadowed_closure();
    //~^ redundant_closure_call

    let shadowed_closure = || 2;
    i = shadowed_closure();
    //~^ redundant_closure_call

    // don't lint here
    let shadowed_closure = || 2;
    i = shadowed_closure();
    i = shadowed_closure();

    // Fix FP in #5916
    #[expect(unused_variables)]
    let mut x;
    let create = || 2 * 2;
    x = create();
    fun(move || {
        x = create();
    })
}

fn fun<T: 'static + FnMut()>(mut f: T) {
    f();
}
