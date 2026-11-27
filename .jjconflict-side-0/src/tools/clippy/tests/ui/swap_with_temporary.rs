#![warn(clippy::swap_with_temporary)]

use std::mem::swap;

fn func() -> String {
    String::from("func")
}

fn func_returning_refmut(s: &mut String) -> &mut String {
    s
}

fn main() {
    let mut x = String::from("x");
    let mut y = String::from("y");
    let mut zz = String::from("zz");
    let z = &mut zz;

    // No lint
    swap(&mut x, &mut y);

    swap(&mut func(), &mut y);
    //~^ ERROR: swapping with a temporary value is inefficient

    swap(&mut x, &mut func());
    //~^ ERROR: swapping with a temporary value is inefficient

    swap(z, &mut func());
    //~^ ERROR: swapping with a temporary value is inefficient

    // No lint
    swap(z, func_returning_refmut(&mut x));

    swap(&mut y, z);

    swap(&mut func(), z);
    //~^ ERROR: swapping with a temporary value is inefficient

    macro_rules! mac {
        (refmut $x:expr) => {
            &mut $x
        };
        (funcall $f:ident) => {
            $f()
        };
        (wholeexpr) => {
            swap(&mut 42, &mut 0)
        };
        (ident $v:ident) => {
            $v
        };
    }
    swap(&mut mac!(funcall func), z);
    //~^ ERROR: swapping with a temporary value is inefficient
    swap(&mut mac!(funcall func), mac!(ident z));
    //~^ ERROR: swapping with a temporary value is inefficient
    swap(mac!(ident z), &mut mac!(funcall func));
    //~^ ERROR: swapping with a temporary value is inefficient
    swap(mac!(refmut y), &mut func());
    //~^ ERROR: swapping with a temporary value is inefficient

    // No lint if it comes from a macro as it may depend on the arguments
    mac!(wholeexpr);
}

struct S {
    t: String,
}

fn dont_lint_those(s: &mut S, v: &mut [String], w: Option<&mut String>) {
    swap(&mut s.t, &mut v[0]);
    swap(&mut s.t, v.get_mut(0).unwrap());
    swap(w.unwrap(), &mut s.t);
}

fn issue15166() {
    use std::sync::Mutex;

    struct A {
        thing: Mutex<Vec<u8>>,
    }

    impl A {
        fn a(&self) {
            let mut new_vec = vec![42];
            // Do not lint here, as neither `new_vec` nor the result of `.lock().unwrap()` are temporaries
            swap(&mut new_vec, &mut self.thing.lock().unwrap());
            for v in new_vec {
                // Do something with v
            }
            // Here `vec![42]` is temporary though, and a proper dereference will have to be used in the fix
            swap(&mut vec![42], &mut self.thing.lock().unwrap());
            //~^ ERROR: swapping with a temporary value is inefficient
        }
    }
}

fn multiple_deref() {
    let mut v1 = &mut &mut &mut vec![42];
    swap(&mut ***v1, &mut vec![]);
    //~^ ERROR: swapping with a temporary value is inefficient

    struct Wrapper<T: ?Sized>(T);
    impl<T: ?Sized> std::ops::Deref for Wrapper<T> {
        type Target = T;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<T: ?Sized> std::ops::DerefMut for Wrapper<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    use std::sync::Mutex;
    let mut v1 = Mutex::new(Wrapper(Wrapper(vec![42])));
    swap(&mut vec![], &mut v1.lock().unwrap());
    //~^ ERROR: swapping with a temporary value is inefficient
}
