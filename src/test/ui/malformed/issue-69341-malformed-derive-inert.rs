fn main() {}

struct CLI {
    #[derive(parse())]
    //~^ ERROR traits in `#[derive(...)]` don't accept arguments
    //~| ERROR cannot find derive macro `parse` in this scope
    path: (),
    //~^ ERROR `derive` may only be applied to structs, enums and unions
}
