//! Regression test for https://github.com/rust-lang/rust/issues/78613.
//! A call argument that needs one more reference should suggest borrowing it.

//@ run-rustfix

fn takes_nested_ref(_: &&str) {}

fn takes_generic_nested_ref<T>(_: &&T) {}

fn takes_nested_mut_ref(_: &mut &str) {}

fn takes_ref(_: &i32) {}

struct Foo;

fn takes_foo_ref(_: &Foo) {}

fn main() {
    let haystack = [&["A1", "A2"][..], &["B1", "B2"], &["C1", "C2"]];
    let needle: &[&str] = &["D1", "D2"];
    let _ = haystack.contains(needle);
    //~^ ERROR mismatched types

    let text = "text";
    takes_nested_ref(text);
    //~^ ERROR mismatched types

    let number = &1;
    takes_generic_nested_ref(number);
    //~^ ERROR mismatched types

    let mut mutable_text = text;
    takes_nested_mut_ref(mutable_text);
    //~^ ERROR mismatched types

    takes_ref(if true { 1 } else { 2 });
    //~^ ERROR mismatched types
    //~| ERROR mismatched types

    // Ordinary `T` to `&T` cases should retain more specific suggestions from the existing path.
    let ref opt = Some(Foo);
    opt.map(|arg| takes_foo_ref(arg));
    //~^ ERROR mismatched types
}
