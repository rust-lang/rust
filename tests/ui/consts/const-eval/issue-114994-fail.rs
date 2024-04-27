// This checks that function pointer signatures that are referenced mutably
// but contain a &mut T parameter still fail in a constant context: see issue #114994.
//
//@ check-fail

const fn use_mut_const_fn(_f: &mut fn(&mut String)) { //~ ERROR mutable references are not allowed in constant functions
    ()
}

const fn use_mut_const_tuple_fn(_f: (fn(), &mut u32)) { //~ ERROR mutable references are not allowed in constant functions

}

fn main() {}
