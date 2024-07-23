#[derive(
    core::RustcDecodable, //~ ERROR cannot find macro `RustcDecodable` in crate `core`
                          //~| ERROR cannot find macro `RustcDecodable` in crate `core`
    core::RustcDecodable, //~ ERROR cannot find macro `RustcDecodable` in crate `core`
                          //~| ERROR cannot find macro `RustcDecodable` in crate `core`
)]
#[core::bench] //~ ERROR cannot find macro `bench` in crate `core`
#[core::global_allocator] //~ ERROR cannot find macro `global_allocator` in crate `core`
#[core::test_case] //~ ERROR cannot find macro `test_case` in crate `core`
#[core::test] //~ ERROR cannot find macro `test` in crate `core`
struct Core;

#[derive(
    std::RustcDecodable, //~ ERROR cannot find macro `RustcDecodable` in crate `std`
                         //~| ERROR cannot find macro `RustcDecodable` in crate `std`
    std::RustcDecodable, //~ ERROR cannot find macro `RustcDecodable` in crate `std`
                         //~| ERROR cannot find macro `RustcDecodable` in crate `std`
)]
#[std::bench] //~ ERROR cannot find macro `bench` in crate `std`
#[std::global_allocator] //~ ERROR cannot find macro `global_allocator` in crate `std`
#[std::test_case] //~ ERROR cannot find macro `test_case` in crate `std`
#[std::test] //~ ERROR cannot find macro `test` in crate `std`
struct Std;

fn main() {}
