//@ check-pass

use std::ffi::VaArgSafe;

unsafe extern "C" fn variadic(_: ...) {}

fn main() {
    unsafe {
        variadic();
        variadic(1_i32);
        variadic(String::new());
        variadic(1_i32, String::new());
        variadic(String::new(), 1_i32);
        variadic(String::new(), String::new());
    }
}

fn generic<T>(x: T) {
    unsafe {
        variadic(x);
    }
}

fn generic_with_bound<T: VaArgSafe>(x: T) {
    unsafe {
        variadic(x);
    }
}

fn indirect() {
    unsafe {
        let f = variadic;
        f(String::new());
        let f_ref = &f;
        f_ref(String::new());
        let g = variadic as unsafe extern "C" fn(...);
        g(String::new());
        let g_ref = &g;
        g_ref(String::new());
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MyStruct {
    x: i32,
    y: i32,
}

unsafe extern "C" fn variadic_after_struct(_: MyStruct, _: ...) {}

fn simple_variadic_after_struct(my_struct: MyStruct) {
    unsafe {
        variadic_after_struct(my_struct);
        variadic_after_struct(my_struct, 1_i32);
        variadic_after_struct(my_struct, String::new());
        variadic_after_struct(my_struct, 1_i32, String::new());
        variadic_after_struct(my_struct, String::new(), 1_i32);
        variadic_after_struct(my_struct, String::new(), String::new());
    }
}

trait Trait {
    type Assoc<'a>;
}

// Unlikely case which our lint doesn't catch.
fn lifetime_dependent<'a, 'b, T: Trait<Assoc<'a>: VaArgSafe>>(x: <T as Trait>::Assoc<'b>) {
    unsafe {
        variadic(x);
    }
}

// We don't lint (thin) references even though they currently don't implement VaArgSafe
fn references<T, U: ?Sized>(tr: &T, tm: &mut T, ur: &U, um: &mut U) {
    unsafe {
        variadic(&String::new());
        variadic(&mut String::new());
        variadic(&String::new() as &dyn Send);
        variadic(&mut String::new() as &mut dyn Send);
        variadic(tr);
        variadic(tm);
        variadic(ur);
        variadic(um);
    }
}

// Quirk with our current hard error: It allows infer vars as varargs
// even if they wouldn't be allowed when the concrete type is known.
fn infer_var() {
    unsafe {
        let mut x = 1;
        variadic(x);
        x = 1_u8;
    }
}

fn integer_float_fallback() {
    unsafe {
        variadic(1);
        variadic(1.0);
    }
}

struct Thing;
impl Thing {
    unsafe extern "C" fn variadic_method(&self, _: ...) {}
}

fn method_call_syntax() {
    unsafe {
        Thing.variadic_method();
        Thing.variadic_method(1_i32);
        Thing.variadic_method(String::new());
        Thing.variadic_method(1_i32, String::new());
        Thing.variadic_method(String::new(), 1_i32);
        Thing.variadic_method(String::new(), String::new());
    }
}
