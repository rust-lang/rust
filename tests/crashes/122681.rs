//@ known-bug: #122681
#[rustc_layout_scalar_valid_range_start(1)]
struct UnitStruct;

#[derive(Default)]
enum SomeEnum {
    #[default]
    Unit,
    Tuple(UnitStruct),
}
