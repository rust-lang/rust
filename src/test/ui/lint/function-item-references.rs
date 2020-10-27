// check-pass
#![feature(c_variadic)]
#![warn(function_item_references)]
use std::fmt::Pointer;
use std::fmt::Formatter;

fn nop() { }
fn foo() -> u32 { 42 }
fn bar(x: u32) -> u32 { x }
fn baz(x: u32, y: u32) -> u32 { x + y }
unsafe fn unsafe_fn() { }
extern "C" fn c_fn() { }
unsafe extern "C" fn unsafe_c_fn() { }
unsafe extern fn variadic(_x: u32, _args: ...) { }

//function references passed to these functions should never lint
fn call_fn(f: &dyn Fn(u32) -> u32, x: u32) { f(x); }
fn parameterized_call_fn<F: Fn(u32) -> u32>(f: &F, x: u32) { f(x); }

//function references passed to these functions should lint
fn print_ptr<F: Pointer>(f: F) { println!("{:p}", f); }
fn bound_by_ptr_trait<F: Pointer>(_f: F) { }
fn bound_by_ptr_trait_tuple<F: Pointer, G: Pointer>(_t: (F, G)) { }
fn implicit_ptr_trait<F>(f: &F) { println!("{:p}", f); }

//case found in tinyvec that triggered a compiler error in an earlier version of the lint checker
trait HasItem {
  type Item;
  fn assoc_item(&self) -> Self::Item;
}
fn _format_assoc_item<T: HasItem>(data: T, f: &mut Formatter) -> std::fmt::Result
    where T::Item: Pointer {
    //when the arg type bound by `Pointer` is an associated type, we shouldn't attempt to normalize
    Pointer::fmt(&data.assoc_item(), f)
}

//simple test to make sure that calls to `Pointer::fmt` aren't double counted
fn _call_pointer_fmt(f: &mut Formatter) -> std::fmt::Result {
    let zst_ref = &foo;
    Pointer::fmt(&zst_ref, f)
    //~^ WARNING taking a reference to a function item does not give a function pointer
}

fn main() {
    //`let` bindings with function references shouldn't lint
    let _ = &foo;
    let _ = &mut foo;

    let zst_ref = &foo;
    let fn_item = foo;
    let indirect_ref = &fn_item;

    let _mut_zst_ref = &mut foo;
    let mut mut_fn_item = foo;
    let _mut_indirect_ref = &mut mut_fn_item;

    let cast_zst_ptr = &foo as *const _;
    let coerced_zst_ptr: *const _ = &foo;

    let _mut_cast_zst_ptr = &mut foo as *mut _;
    let _mut_coerced_zst_ptr: *mut _ = &mut foo;

    let _cast_zst_ref = &foo as &dyn Fn() -> u32;
    let _coerced_zst_ref: &dyn Fn() -> u32 = &foo;

    let _mut_cast_zst_ref = &mut foo as &mut dyn Fn() -> u32;
    let _mut_coerced_zst_ref: &mut dyn Fn() -> u32 = &mut foo;

    //the suggested way to cast to a function pointer
    let fn_ptr = foo as fn() -> u32;

    //correct ways to print function pointers
    println!("{:p}", foo as fn() -> u32);
    println!("{:p}", fn_ptr);

    //potential ways to incorrectly try printing function pointers
    println!("{:p}", &foo);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    print!("{:p}", &foo);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    format!("{:p}", &foo);
    //~^ WARNING taking a reference to a function item does not give a function pointer

    println!("{:p}", &foo as *const _);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    println!("{:p}", zst_ref);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    println!("{:p}", cast_zst_ptr);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    println!("{:p}", coerced_zst_ptr);
    //~^ WARNING taking a reference to a function item does not give a function pointer

    println!("{:p}", &fn_item);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    println!("{:p}", indirect_ref);
    //~^ WARNING taking a reference to a function item does not give a function pointer

    println!("{:p}", &nop);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    println!("{:p}", &bar);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    println!("{:p}", &baz);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    println!("{:p}", &unsafe_fn);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    println!("{:p}", &c_fn);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    println!("{:p}", &unsafe_c_fn);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    println!("{:p}", &variadic);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    println!("{:p}", &std::env::var::<String>);
    //~^ WARNING taking a reference to a function item does not give a function pointer

    println!("{:p} {:p} {:p}", &nop, &foo, &bar);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    //~^^ WARNING taking a reference to a function item does not give a function pointer
    //~^^^ WARNING taking a reference to a function item does not give a function pointer

    //using a function reference to call a function shouldn't lint
    (&bar)(1);

    //passing a function reference to an arbitrary function shouldn't lint
    call_fn(&bar, 1);
    parameterized_call_fn(&bar, 1);
    std::mem::size_of_val(&foo);

    unsafe {
        //potential ways to incorrectly try transmuting function pointers
        std::mem::transmute::<_, usize>(&foo);
        //~^ WARNING taking a reference to a function item does not give a function pointer
        std::mem::transmute::<_, (usize, usize)>((&foo, &bar));
        //~^ WARNING taking a reference to a function item does not give a function pointer
        //~^^ WARNING taking a reference to a function item does not give a function pointer

        //the correct way to transmute function pointers
        std::mem::transmute::<_, usize>(foo as fn() -> u32);
        std::mem::transmute::<_, (usize, usize)>((foo as fn() -> u32, bar as fn(u32) -> u32));
    }

    //function references as arguments required to be bound by std::fmt::Pointer should lint
    print_ptr(&bar);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    bound_by_ptr_trait(&bar);
    //~^ WARNING taking a reference to a function item does not give a function pointer
    bound_by_ptr_trait_tuple((&foo, &bar));
    //~^ WARNING taking a reference to a function item does not give a function pointer
    //~^^ WARNING taking a reference to a function item does not give a function pointer
    implicit_ptr_trait(&bar); // ignore

    //correct ways to pass function pointers as arguments bound by std::fmt::Pointer
    print_ptr(bar as fn(u32) -> u32);
    bound_by_ptr_trait(bar as fn(u32) -> u32);
    bound_by_ptr_trait_tuple((foo as fn() -> u32, bar as fn(u32) -> u32));
}
