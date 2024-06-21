#[derive(Send)]
//~^ ERROR cannot find derive macro `Send`
//~| ERROR cannot find derive macro `Send`
struct Test;

#[derive(Sync)]
//~^ ERROR cannot find derive macro `Sync`
//~| ERROR cannot find derive macro `Sync`
struct Test1;

pub fn main() {}
