#[derive(
    core::RustcDecodable, //~ ERROR cannot find macro `RustcDecodable`
                          //~| ERROR cannot find macro `RustcDecodable`
    core::RustcDecodable, //~ ERROR cannot find macro `RustcDecodable`
                          //~| ERROR cannot find macro `RustcDecodable`
)]
#[core::bench] //~ ERROR cannot find macro `bench`
#[core::global_allocator] //~ ERROR cannot find macro `global_allocator`
#[core::test_case] //~ ERROR cannot find macro `test_case`
#[core::test] //~ ERROR cannot find macro `test`
struct Core;

#[derive(
    std::RustcDecodable, //~ ERROR cannot find macro `RustcDecodable`
                         //~| ERROR cannot find macro `RustcDecodable`
    std::RustcDecodable, //~ ERROR cannot find macro `RustcDecodable`
                         //~| ERROR cannot find macro `RustcDecodable`
)]
#[std::bench] //~ ERROR cannot find macro `bench`
#[std::global_allocator] //~ ERROR cannot find macro `global_allocator`
#[std::test_case] //~ ERROR cannot find macro `test_case`
#[std::test] //~ ERROR cannot find macro `test`
struct Std;

fn main() {}
