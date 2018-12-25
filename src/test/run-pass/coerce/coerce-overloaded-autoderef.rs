// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

use std::rc::Rc;

// Examples from the "deref coercions" RFC, at rust-lang/rfcs#241.

fn use_ref<T>(_: &T) {}
fn use_mut<T>(_: &mut T) {}

fn use_rc<T>(t: Rc<T>) {
    use_ref(&*t);  // what you have to write today
    use_ref(&t);   // what you'd be able to write
    use_ref(&&&&&&t);
    use_ref(&mut &&&&&t);
    use_ref(&&&mut &&&t);
}

fn use_mut_box<T>(mut t: &mut Box<T>) {
    use_mut(&mut *t); // what you have to write today
    use_mut(t);       // what you'd be able to write
    use_mut(&mut &mut &mut t);

    use_ref(&*t);      // what you have to write today
    use_ref(t);        // what you'd be able to write
    use_ref(&&&&&&t);
    use_ref(&mut &&&&&t);
    use_ref(&&&mut &&&t);
}

fn use_nested<T>(t: &Box<T>) {
    use_ref(&**t);  // what you have to write today
    use_ref(t);     // what you'd be able to write (note: recursive deref)
    use_ref(&&&&&&t);
    use_ref(&mut &&&&&t);
    use_ref(&&&mut &&&t);
}

fn use_slice(_: &[u8]) {}
fn use_slice_mut(_: &mut [u8]) {}

fn use_vec(mut v: Vec<u8>) {
    use_slice_mut(&mut v[..]); // what you have to write today
    use_slice_mut(&mut v);     // what you'd be able to write
    use_slice_mut(&mut &mut &mut v);

    use_slice(&v[..]);  // what you have to write today
    use_slice(&v);      // what you'd be able to write
    use_slice(&&&&&&v);
    use_slice(&mut &&&&&v);
    use_slice(&&&mut &&&v);
}

fn use_vec_ref(v: &Vec<u8>) {
    use_slice(&v[..]);  // what you have to write today
    use_slice(v);       // what you'd be able to write
    use_slice(&&&&&&v);
    use_slice(&mut &&&&&v);
    use_slice(&&&mut &&&v);
}

fn use_op_rhs(s: &mut String) {
    *s += {&String::from(" ")};
}

pub fn main() {}
