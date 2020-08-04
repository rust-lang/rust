//~ ERROR failed to resolve: could not find `future` in `std` [E0433]
//~^ ERROR failed to resolve: could not find `pin` in `std` [E0433]
//~^^ ERROR failed to resolve: could not find `future` in `std` [E0433]
//~^^^ ERROR failed to resolve: could not find `future` in `std` [E0433]
//~^^^^ ERROR failed to resolve: could not find `task` in `std` [E0433]
//~^^^^^ ERROR failed to resolve: could not find `task` in `std` [E0433]
//~^^^^^^ ERROR failed to resolve: could not find `future` in `std` [E0433]
//~^^^^^^^ ERROR failed to resolve: could not find `future` in `std` [E0433]
//~^^^^^^^^ ERROR failed to resolve: could not find `ops` in `std` [E0433]
//~^^^^^^^^^ ERROR failed to resolve: could not find `option` in `std` [E0433]
//~^^^^^^^^^^ ERROR failed to resolve: could not find `option` in `std` [E0433]
//~^^^^^^^^^^^ ERROR failed to resolve: could not find `iter` in `std` [E0433]
//~^^^^^^^^^^^^ ERROR failed to resolve: could not find `iter` in `std` [E0433]
//~^^^^^^^^^^^^^ ERROR failed to resolve: could not find `ops` in `std` [E0433]
//~^^^^^^^^^^^^^^ ERROR failed to resolve: could not find `option` in `std` [E0433]
//~^^^^^^^^^^^^^^^ ERROR failed to resolve: could not find `option` in `std` [E0433]
//~^^^^^^^^^^^^^^^^ ERROR failed to resolve: could not find `iter` in `std` [E0433]
//~^^^^^^^^^^^^^^^^^ ERROR failed to resolve: could not find `iter` in `std` [E0433]
//~^^^^^^^^^^^^^^^^^^ ERROR failed to resolve: could not find `ops` in `std` [E0433]
//~^^^^^^^^^^^^^^^^^^^ ERROR failed to resolve: could not find `result` in `std` [E0433]
//~^^^^^^^^^^^^^^^^^^^^ ERROR failed to resolve: could not find `convert` in `std` [E0433]
//~^^^^^^^^^^^^^^^^^^^^^ ERROR failed to resolve: could not find `ops` in `std` [E0433]
//~^^^^^^^^^^^^^^^^^^^^^^ ERROR failed to resolve: could not find `result` in `std` [E0433]

// edition:2018
// aux-build:not-libstd.rs

// Check that paths created in HIR are not affected by in scope names.

extern crate not_libstd as std;

async fn the_future() {
    async {}.await;
}

fn main() -> Result<(), ()> {
    for i in 0..10 {}
    for j in 0..=10 {}
    Ok(())?;
    Ok(())
}
