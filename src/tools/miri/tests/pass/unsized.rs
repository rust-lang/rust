//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(unsized_fn_params)]
#![feature(custom_mir, core_intrinsics)]

fn unsized_params() {
    pub fn f0(_f: dyn FnOnce()) {}
    pub fn f1(_s: str) {}
    pub fn f2(_x: i32, _y: [i32]) {}
    pub fn f3(_p: dyn Send) {}

    let c: Box<dyn FnOnce()> = Box::new(|| {});
    f0(*c);
    let foo = "foo".to_string().into_boxed_str();
    f1(*foo);
    let sl: Box<[i32]> = [0, 1, 2].to_vec().into_boxed_slice();
    f2(5, *sl);
    let p: Box<dyn Send> = Box::new((1, 2));
    f3(*p);
}

fn unsized_field_projection() {
    use std::intrinsics::mir::*;

    pub struct S<T: ?Sized>(T);

    #[custom_mir(dialect = "runtime", phase = "optimized")]
    fn f(x: S<[u8]>) {
        mir! {
            {
                let idx = 0;
                // Project to an unsized field of an unsized local.
                x.0[idx] = 0;
                let _val = x.0[idx];
                Return()
            }
        }
    }

    let x: Box<S<[u8]>> = Box::new(S([0]));
    f(*x);
}

fn main() {
    unsized_params();
    unsized_field_projection();
}
