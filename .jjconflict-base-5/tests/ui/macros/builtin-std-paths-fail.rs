#[derive(
    core::RustcDecodable, //~ ERROR cannot find `RustcDecodable` in `core`
                          //~| ERROR cannot find `RustcDecodable` in `core`
    core::RustcDecodable, //~ ERROR cannot find `RustcDecodable` in `core`
                          //~| ERROR cannot find `RustcDecodable` in `core`
)]
#[core::bench] //~ ERROR cannot find `bench` in `core`
#[core::global_allocator] //~ ERROR cannot find `global_allocator` in `core`
#[core::test_case] //~ ERROR cannot find `test_case` in `core`
#[core::test] //~ ERROR cannot find `test` in `core`
struct Core;

#[derive(
    std::RustcDecodable, //~ ERROR cannot find `RustcDecodable` in `std`
                         //~| ERROR cannot find `RustcDecodable` in `std`
    std::RustcDecodable, //~ ERROR cannot find `RustcDecodable` in `std`
                         //~| ERROR cannot find `RustcDecodable` in `std`
)]
#[std::bench] //~ ERROR cannot find `bench` in `std`
#[std::global_allocator] //~ ERROR cannot find `global_allocator` in `std`
#[std::test_case] //~ ERROR cannot find `test_case` in `std`
#[std::test] //~ ERROR cannot find `test` in `std`
struct Std;

fn main() {}
