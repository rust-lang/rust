//@ edition: 2024

use std::fmt::Display;

fn display_len<T>(x: &Vec<T>) -> impl Display {
    //~^ NOTE in this expansion of desugaring of `impl Trait`
    //~| NOTE in this expansion of desugaring of `impl Trait`
    //~| NOTE in this expansion of desugaring of `impl Trait`
    //~| NOTE in this expansion of desugaring of `impl Trait`
    //~| NOTE in this expansion of desugaring of `impl Trait`
    x.len()
}

fn conflicting_borrow() {
    let mut x = vec![];
    let a = display_len(&x);
    //~^ NOTE this call may capture more lifetimes than intended
    //~| NOTE immutable borrow occurs here
    x.push(1);
    //~^ ERROR cannot borrow `x` as mutable because it is also borrowed as immutable
    //~| NOTE mutable borrow occurs here
    println!("{a}");
    //~^ NOTE immutable borrow later used here
}

fn needs_static() {
    let x = vec![1];
    //~^ NOTE binding `x` declared here
    let a = display_len(&x);
    //~^ ERROR `x` does not live long enough
    //~| NOTE this call may capture more lifetimes than intended
    //~| NOTE borrowed value does not live long enoug

    fn needs_static(_: impl Sized + 'static) {}
    //~^ NOTE requirement that the value outlives `'static` introduced here
    needs_static(a);
    //~^ NOTE argument requires that `x` is borrowed for `'static`
}
//~^ NOTE `x` dropped here while still borrowed

fn is_moved() {
    let x = vec![1];
    //~^ NOTE binding `x` declared here
    let a = display_len(&x);
    //~^ NOTE this call may capture more lifetimes than intended
    //~| NOTE borrow of `x` occurs here

    fn mv(_: impl Sized) {}
    mv(x);
    //~^ ERROR cannot move out of `x` because it is borrowed
    //~| NOTE move out of `x` occurs here
}
//~^ NOTE borrow might be used here, when `a` is dropped

fn display_len_mut<T>(x: &mut Vec<T>) -> impl Display {
    //~^ NOTE in this expansion of desugaring of `impl Trait`
    //~| NOTE in this expansion of desugaring of `impl Trait`
    //~| NOTE in this expansion of desugaring of `impl Trait`
    x.len()
}

fn conflicting_borrow_mut() {
    let mut x = vec![];
    let a = display_len_mut(&mut x);
    //~^ NOTE this call may capture more lifetimes than intended
    //~| NOTE first mutable borrow occurs here
    x.push(1);
    //~^ ERROR cannot borrow `x` as mutable more than once
    //~| NOTE second mutable borrow occurs here
    println!("{a}");
    //~^ NOTE first borrow later used here
}

fn needs_static_mut() {
    let mut x = vec![1];
    //~^ NOTE binding `x` declared here
    let a = display_len_mut(&mut x);
    //~^ ERROR `x` does not live long enough
    //~| NOTE this call may capture more lifetimes than intended
    //~| NOTE borrowed value does not live long enough

    fn needs_static(_: impl Sized + 'static) {}
    //~^ NOTE requirement that the value outlives `'static` introduced here
    needs_static(a);
    //~^ NOTE argument requires that `x` is borrowed for `'static`
}
//~^ NOTE `x` dropped here while still borrowed

fn is_move_mut() {
    let mut x = vec![1];
    //~^ NOTE binding `x` declared here
    let a = display_len_mut(&mut x);
    //~^ NOTE this call may capture more lifetimes than intended
    //~| NOTE borrow of `x` occurs here

    fn mv(_: impl Sized) {}
    mv(x);
    //~^ ERROR cannot move out of `x` because it is borrowed
    //~| NOTE move out of `x` occurs here
}
//~^ NOTE borrow might be used here, when `a` is dropped

struct S { f: i32 }

fn display_field<T: Copy + Display>(t: &T) -> impl Display {
    //~^ NOTE in this expansion of desugaring of `impl Trait`
    //~| NOTE in this expansion of desugaring of `impl Trait`
    //~| NOTE in this expansion of desugaring of `impl Trait`
    *t
}

fn conflicting_borrow_field() {
    let mut s = S { f: 0 };
    let a = display_field(&s.f);
    //~^ NOTE this call may capture more lifetimes than intended
    //~| NOTE `s.f` is borrowed here
    s.f = 1;
    //~^ ERROR cannot assign to `s.f` because it is borrowed
    //~| NOTE `s.f` is assigned to here but it was already borrowed
    println!("{a}");
    //~^ NOTE borrow later used here
}

fn display_field_mut<T: Copy + Display>(t: &mut T) -> impl Display {
    *t
}

fn conflicting_borrow_field_mut() {
    let mut s = S { f: 0 };
    let a = display_field(&mut s.f);
    //~^ NOTE this call may capture more lifetimes than intended
    //~| NOTE `s.f` is borrowed here
    s.f = 1;
    //~^ ERROR cannot assign to `s.f` because it is borrowed
    //~| NOTE `s.f` is assigned to here but it was already borrowed
    println!("{a}");
    //~^ NOTE borrow later used here
}

fn field_move() {
    let mut s = S { f: 0 };
    let a = display_field(&mut s.f);
    //~^ NOTE this call may capture more lifetimes than intended
    //~| NOTE `s.f` is borrowed here
    s.f;
    //~^ ERROR cannot use `s.f` because it was mutably borrowed
    //~| NOTE use of borrowed `s.f`
    println!("{a}");
    //~^ NOTE borrow later used here
}

struct Z {
    f: Vec<i32>,
}

fn live_long() {
    let x;
    {
        let z = Z { f: vec![1] };
        //~^ NOTE binding `z` declared here
        x = display_len(&z.f);
        //~^ ERROR `z.f` does not live long enough
        //~| NOTE this call may capture more lifetimes than intended
        //~| NOTE values in a scope are dropped in the opposite order they are defined
        //~| NOTE borrowed value does not live long enough
    }
    //~^ NOTE `z.f` dropped here while still borrowed
}
//~^ NOTE borrow might be used here, when `x` is dropped

fn temp() {
    let x = { let x = display_len(&mut vec![0]); x };
    //~^ ERROR temporary value dropped while borrowed
    //~| NOTE this call may capture more lifetimes than intended
    //~| NOTE consider using a `let` binding to create a longer lived value
    //~| NOTE borrow later used here
    //~| NOTE temporary value is freed at the end of this statement
}

// FIXME: This doesn't display a useful Rust 2024 suggestion :(
fn returned() -> impl Sized {
    let x = vec![0];
    //~^ NOTE binding `x` declared here
    display_len(&x)
    //~^ ERROR `x` does not live long enough
    //~| NOTE borrowed value does not live long enough
    //~| NOTE argument requires that `x` is borrowed for `'static`
}
//~^ NOTE `x` dropped here while still borrowed

fn capture_apit(x: &impl Sized) -> impl Sized {}
//~^ NOTE you could use a `use<...>` bound to explicitly specify captures, but

fn test_apit() {
    let x = String::new();
    //~^ NOTE binding `x` declared here
    let y = capture_apit(&x);
    //~^ NOTE borrow of `x` occurs here
    //~| NOTE this call may capture more lifetimes than intended
    drop(x);
    //~^ ERROR cannot move out of `x` because it is borrowed
    //~| NOTE move out of `x` occurs here
}
//~^ NOTE borrow might be used here, when `y` is dropped

fn main() {}
