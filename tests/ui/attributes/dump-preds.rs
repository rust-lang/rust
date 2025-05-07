//@ normalize-stderr: "DefId\(.+?\)" -> "DefId(..)"

#![feature(rustc_attrs)]

#[rustc_dump_predicates]
trait Trait<T>: Iterator<Item: Copy>
//~^ ERROR rustc_dump_predicates
where
    String: From<T>
{
    #[rustc_dump_predicates]
    #[rustc_dump_item_bounds]
    type Assoc<P: Eq>: std::ops::Deref<Target = ()>
    //~^ ERROR rustc_dump_predicates
    //~| ERROR rustc_dump_item_bounds
    where
        Self::Assoc<()>: Copy;
}

fn main() {}
