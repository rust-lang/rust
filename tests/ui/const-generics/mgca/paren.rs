//@ check-pass
//! See also: tests/ui/const-generics/paren.rs
#![feature(min_generic_const_args, generic_const_items)]

struct Thing<const N: usize>;

type const A<const N: usize>: usize = N;

fn f<const N: usize>() {
    let _: [u32; core::direct_const_arg!(_)] = [5; 5];
    let _: [u32; core::direct_const_arg!((_))] = [5; 5];
    let _: [u32; core::direct_const_arg!({ _ })] = [5; 5];
    let _: [u32; core::direct_const_arg!({ (_) })] = [5; 5];
    let _: [u32; core::direct_const_arg!(N)] = [5; _];
    let _: [u32; core::direct_const_arg!((N))] = [5; _];
    let _: [u32; core::direct_const_arg!({ N })] = [5; _];
    let _: [u32; core::direct_const_arg!({ (N) })] = [5; _];
    let _: [u32; core::direct_const_arg!(A::<N>)] = [5; _];
    let _: [u32; core::direct_const_arg!((A::<N>))] = [5; _];
    let _: [u32; core::direct_const_arg!({ A::<N> })] = [5; _];
    let _: [u32; core::direct_const_arg!({ (A::<N>) })] = [5; _];
    let _: Thing<core::direct_const_arg!(_)> = Thing::<5>;
    let _: Thing<core::direct_const_arg!((_))> = Thing::<5>;
    let _: Thing<core::direct_const_arg!({ _ })> = Thing::<5>;
    let _: Thing<core::direct_const_arg!({ (_) })> = Thing::<5>;
    let _: Thing<core::direct_const_arg!(N)> = Thing;
    let _: Thing<core::direct_const_arg!((N))> = Thing;
    let _: Thing<core::direct_const_arg!({ N })> = Thing;
    let _: Thing<core::direct_const_arg!({ (N) })> = Thing;
    let _: Thing<core::direct_const_arg!(A::<N>)> = Thing;
    let _: Thing<core::direct_const_arg!((A::<N>))> = Thing;
    let _: Thing<core::direct_const_arg!({ A::<N> })> = Thing;
    let _: Thing<core::direct_const_arg!({ (A::<N>) })> = Thing;
}

fn main() {}
