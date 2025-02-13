// See `stable_order_of_exportable_impls` doc.
#![feature(export, rustc_attrs)]

struct S;

#[export]
#[rustc_stable_impl_id]
impl S {}
//~^ ERROR: stable impl id: 0

#[export]
mod m1 {
    #[rustc_stable_impl_id]
    impl crate::S {}
    //~^ ERROR: stable impl id: 1
    #[rustc_stable_impl_id]
    impl crate::S {}
    //~^ ERROR: stable impl id: 2
    mod m2 {
        #[rustc_stable_impl_id]
        impl crate::S {}
        //~^ ERROR: stable impl id: 3
    }
}

#[export]
#[rustc_stable_impl_id]
impl S {}
//~^ ERROR: stable impl id: 4

#[export]
mod m3 {
    #[rustc_stable_impl_id]
    impl crate::S {}
    //~^ ERROR: stable impl id: 5
}

fn main() {}
